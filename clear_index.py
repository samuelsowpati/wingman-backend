#!/usr/bin/env python3
"""
Script to delete all vectors from the Pinecone index.
Use this to start fresh when you've fixed the PDF processing.
"""

import asyncio
from dotenv import load_dotenv
from vector_store import vector_store

async def clear_pinecone_index():
    """Delete all vectors from the Pinecone index."""
    
    print("🗑️  Clearing Pinecone Index")
    print("=" * 40)
    
    try:
        # Get current stats
        print("1. 📊 Checking current index stats...")
        stats = await vector_store.get_index_stats()
        total_vectors = stats.get('total_vectors', 0)
        print(f"   Current vectors in index: {total_vectors}")
        
        if total_vectors == 0:
            print("✅ Index is already empty!")
            return
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will delete ALL {total_vectors} vectors from the index!")
        print(f"   Index name: {vector_store.index_name}")
        
        # Delete all vectors
        print("\n2. 🗑️  Deleting all vectors...")
        
        # Method 1: Delete all vectors by deleting everything
        # This is the most reliable way to clear everything
        delete_response = vector_store.index.delete(delete_all=True)
        
        print("✅ Delete request sent to Pinecone")
        
        # Wait a moment for the deletion to process
        print("3. ⏳ Waiting for deletion to complete...")
        await asyncio.sleep(3)
        
        # Check if deletion worked
        print("4. 📊 Verifying deletion...")
        new_stats = await vector_store.get_index_stats()
        new_total = new_stats.get('total_vectors', 0)
        
        if new_total == 0:
            print("✅ SUCCESS: Index has been completely cleared!")
        else:
            print(f"⚠️  Index still has {new_total} vectors. It may take a few moments to fully clear.")
            print("   Try checking again in a minute.")
        
        print(f"\n🎯 Next steps:")
        print("1. Run: python main.py (to start your backend)")
        print("2. POST http://localhost:8000/api/batch-process (to reprocess PDFs with the fix)")
        print("3. Test your SAF/AA query again")
        
    except Exception as e:
        print(f"❌ Error clearing index: {str(e)}")
        print("\nTroubleshooting:")
        print("- Make sure your PINECONE_API_KEY is set in .env")
        print("- Check that you have network connectivity")
        print("- Verify your Pinecone account has access to the index")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(clear_pinecone_index())
