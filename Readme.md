# 🧠 ChromaDB Vector Database with Gemini Embeddings

This project demonstrates how to store and manage **text embeddings** in **ChromaDB Cloud** using **Google Gemini**.  
It’s a beginner-friendly tutorial for building AI-powered vector databases.

---
📍 Where Are Vector Databases Used Today?

Vector databases like ChromaDB, Pinecone, and Weaviate are becoming essential in modern AI applications.
They help computers understand meaning (semantics) instead of just exact words.

Here are some real-world examples 👇

🧠 1. AI Chatbots and RAG Systems

Used to store document embeddings for chatbots.

Example: ChatGPT’s “Retrieval-Augmented Generation (RAG)” systems use vector databases to find contextually relevant info.

🔍 2. Semantic Search Engines

Helps find similar results based on meaning, not just keywords.
Example: Searching “car repair manual” may also return “vehicle maintenance guide”.

🖼️ 3. Image and Video Similarity

Used to find visually similar images or videos using embeddings.
Example: “Find products that look like this shirt” in e-commerce.

## 🚀 Features

- Connects securely to **ChromaDB Cloud**
- Uses **Gemini API** for embedding generation
- Saves, stores, and retrieves text embeddings
- Environment-based configuration for credentials
- Simple, clean, and extensible Python code

---

## 🧩 Project Structure
```
📁 chromadb-gemini-demo
├── main.py # Core script for embeddings and saving to Chroma
├── requirements.txt # Python dependencies
├── .env # Environment variables
├── data.txt # Input text file
├── setup.txt # Step-by-step setup guide
└── README.md # Documentation
