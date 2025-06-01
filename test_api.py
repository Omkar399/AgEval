#!/usr/bin/env python3
"""
Quick test of the Hard Thinking API endpoint
"""

import asyncio
import json
from fastapi.testclient import TestClient
from fastapi_app import app

def test_hardthinking_api():
    """Test the /api/hardthinking/run endpoint"""
    
    client = TestClient(app)
    
    # Test data
    test_payload = {
        "query": "What is 2 + 2?",
        "problem_type": "math",
        "complexity": "simple",
        "strategy": "voting"
    }
    
    print("🧪 Testing Hard Thinking API Endpoint")
    print("=" * 50)
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        # Make the API call
        response = client.post("/api/hardthinking/run", json=test_payload)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success Response:")
            print(json.dumps(result, indent=2))
            
            if result.get('status') == 'completed' and 'result' in result:
                print(f"\n🎯 Hard Thinking Result Summary:")
                hard_result = result['result']
                print(f"   Best Model: {hard_result.get('best_model')}")
                print(f"   Confidence: {hard_result.get('confidence_score', 0):.3f}")
                print(f"   Consensus: {hard_result.get('consensus_level', 0):.1f}%")
                print(f"   Processing Time: {hard_result.get('processing_time', 0):.2f}s")
                print(f"   Total Tokens: {hard_result.get('total_tokens', 0):,}")
                print(f"   Decomposition Steps: {len(hard_result.get('decomposition', []))}")
                
        else:
            print(f"\n❌ Error Response:")
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2))
            except:
                print(f"Raw response: {response.text}")
                
    except Exception as e:
        print(f"\n💥 Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hardthinking_api()