
import os
import re
import numpy as np
import torch
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModel, pipeline
from huggingface_hub import login
import chromadb
import faiss

class YouTubePlaylistRAG:
    def __init__(self, playlist_url):
        self.playlist_url = playlist_url
        self.transcripts = {}
        self.all_chunks = []
        self.metadata = []
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.qa_model_name = "gpt2"
        self.chroma_path = "./chroma_db"
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name)
        self.generator = None
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="youtube_rag")

    def fetch_transcripts(self):
        """Fetch transcripts for all videos in the playlist"""
        playlist = Playlist(self.playlist_url)
        print(f"Found {len(list(playlist.video_urls))} videos in the playlist.")
        
        for video in playlist.videos:
            video_id = video.video_id
            try:
                transcript_segments = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join(segment['text'] for segment in transcript_segments)
                self.transcripts[video_id] = full_text
                print(f"Transcript retrieved for video ID: {video_id}")
            except Exception as e:
                print(f"Could not retrieve transcript for video ID {video_id}: {e}")

    @staticmethod
    def preprocess_text(text):
        """Clean text by removing extra whitespace"""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def chunk_text(text, chunk_size=200, overlap=20):
        """Split text into chunks with specified size and overlap"""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk = " ".join(words[start:start + chunk_size])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def process_transcripts(self):
        """Process all transcripts into chunks"""
        for vid, text in self.transcripts.items():
            clean_text = self.preprocess_text(text)
            chunks = self.chunk_text(clean_text)
            for i, chunk in enumerate(chunks):
                self.all_chunks.append(chunk)
                self.metadata.append({"video_id": vid, "chunk_index": i})
        print(f"Total chunks generated: {len(self.all_chunks)}")

    def get_embedding(self, text):
        """Generate embedding for text using mean pooling"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def generate_embeddings(self):
        """Generate embeddings for all chunks"""
        self.embeddings = [self.get_embedding(chunk) for chunk in self.all_chunks]
        self.embedding_matrix = np.vstack(self.embeddings).astype("float32")
        print(f"Embeddings generated with dimension: {self.embedding_matrix.shape[1]}")

    def setup_faiss(self):
        """Create FAISS index for similarity search"""
        self.index = faiss.IndexFlatL2(self.embedding_matrix.shape[1])
        self.index.add(self.embedding_matrix)
        print(f"FAISS index created with {self.index.ntotal} embeddings")

    def store_in_chroma(self):
        """Store documents and embeddings in ChromaDB"""
        for idx, (chunk, embedding) in enumerate(zip(self.all_chunks, self.embeddings)):
            self.collection.add(
                ids=[f"doc_{idx}"],
                documents=[chunk],
                embeddings=[embedding.tolist()]
            )
        print("Embeddings stored successfully in ChromaDB!")

    def initialize_generator(self, hf_token):
        """Initialize the text generation pipeline"""
        login(token=hf_token)
        self.generator = pipeline("text-generation", model=self.qa_model_name)

    def retrieve_documents(self, query, top_k=3):
        """Retrieve relevant documents using ChromaDB"""
        query_embedding = self.get_embedding(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0], results["ids"][0]

    def generate_answer(self, query, context):
        """Generate answer using LLM"""
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        return self.generator(prompt, max_new_tokens=100)[0]['generated_text']

    def interactive_loop(self):
        """Run interactive question answering loop"""
        print("\n=== RAG System Ready ===")
        while True:
            query = input("\nAsk a question (or 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                print("Exiting...")
                break
            
            documents, sources = self.retrieve_documents(query)
            context = " ".join(documents)
            
            answer = self.generate_answer(query, context)
            
            print("\n🤖 Answer:")
            print(answer.split("Answer:")[-1].strip())
            print("\n🔍 Sources:")
            for idx, source in enumerate(sources, 1):
                print(f"{idx}. {source}")

if __name__ == "__main__":
    # Configuration
    PLAYLIST_URL = "https://www.youtube.com/playlist?list=PL8PYTP1V4I8D4BeyjwWczukWq9d8PNyZp"
    HF_TOKEN = "your_huggingface_token_here"  # Replace with your token
    
    # Initialize RAG system
    rag_system = YouTubePlaylistRAG(PLAYLIST_URL)
    
    # Build knowledge base
    rag_system.fetch_transcripts()
    rag_system.process_transcripts()
    rag_system.generate_embeddings()
    rag_system.setup_faiss()
    rag_system.store_in_chroma()
    
    # Initialize QA system
    rag_system.initialize_generator(HF_TOKEN)
    
    # Start interactive session
    rag_system.interactive_loop()
