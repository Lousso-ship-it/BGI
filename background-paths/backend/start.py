#!/usr/bin/env python3
"""
Script de démarrage pour le backend FastAPI
"""
import uvicorn
import os

if __name__ == "__main__":
    # Change to the root directory to access CSV files
    os.chdir("..")
    
    # Run the FastAPI server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
        log_level="info"
    )
