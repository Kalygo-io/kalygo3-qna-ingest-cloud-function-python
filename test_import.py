#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""
try:
    print("Testing imports...")
    from main import process_qna_ingest_topic_message
    print("✓ main.py imported successfully")
    print(f"✓ Function found: {process_qna_ingest_topic_message.__name__}")
    
    from helpers import gcs, csv_processor, embedding, get_secret, pinecone
    print("✓ All helper modules imported successfully")
    
    from clients import gcs_client_factory, secret_manager_client
    print("✓ All client modules imported successfully")
    
    from singletons import environment_variables
    print("✓ All singleton modules imported successfully")
    
    print("\n✅ All imports successful! Function should deploy correctly.")
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

