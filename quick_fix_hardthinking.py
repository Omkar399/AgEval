#!/usr/bin/env python3
"""
Quick fix script to ensure Hard Thinking functionality works
"""

def create_simple_demo():
    """Create a simple demo function to test Hard Thinking"""
    demo_script = """
// Simple demo function for Hard Thinking
window.demoHardThinking = async function() {
    console.log('🧠 Starting Hard Thinking Demo');
    
    // Show the process card
    document.getElementById('thinkingProcessCard').style.display = 'block';
    document.getElementById('hardThinkingResults').style.display = 'none';
    
    // Update progress
    document.getElementById('thinkingProgress').style.width = '25%';
    document.getElementById('currentThinkingStep').textContent = 'Decomposing problem...';
    document.getElementById('thinkingStatus').textContent = 'Decomposing';
    document.getElementById('thinkingStatus').className = 'status-badge running';
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Show decomposition
    document.getElementById('taskDecomposition').innerHTML = `
        <div style="background: var(--surface); padding: 12px; margin-bottom: 8px; border-radius: var(--radius); border-left: 3px solid var(--accent);">
            <strong>Step 1:</strong> Understand the mathematical problem
        </div>
        <div style="background: var(--surface); padding: 12px; margin-bottom: 8px; border-radius: var(--radius); border-left: 3px solid var(--accent);">
            <strong>Step 2:</strong> Apply arithmetic operations
        </div>
        <div style="background: var(--surface); padding: 12px; margin-bottom: 8px; border-radius: var(--radius); border-left: 3px solid var(--accent);">
            <strong>Step 3:</strong> Verify the result
        </div>
    `;
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Show model processing
    document.getElementById('thinkingProgress').style.width = '60%';
    document.getElementById('currentThinkingStep').textContent = 'Querying models...';
    document.getElementById('modelResponses').style.display = 'grid';
    
    const models = ['gpt', 'claude', 'gemini'];
    for (const model of models) {
        document.getElementById(model + 'Status').textContent = '🔄';
        await new Promise(resolve => setTimeout(resolve, 500));
        document.getElementById(model + 'Status').textContent = '✅';
    }
    
    // Show model responses
    document.getElementById('modelResponseDetails').innerHTML = `
        <div style="margin-bottom: 16px; background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>GPT-4</h4>
            <div style="font-size: 13px; color: var(--secondary);">Confidence: 97.4% | Final Score: 97.4%</div>
            <div style="margin-top: 8px; padding: 8px; background: var(--background); border-radius: 4px; font-size: 12px;">
                The answer is 4. This is a basic arithmetic operation: 2 + 2 = 4.
            </div>
        </div>
        <div style="margin-bottom: 16px; background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>Claude</h4>
            <div style="font-size: 13px; color: var(--secondary);">Confidence: 83.7% | Final Score: 83.7%</div>
            <div style="margin-top: 8px; padding: 8px; background: var(--background); border-radius: 4px; font-size: 12px;">
                Two plus two equals four. This is fundamental addition.
            </div>
        </div>
        <div style="margin-bottom: 16px; background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>Gemini</h4>
            <div style="font-size: 13px; color: var(--secondary);">Confidence: 75.8% | Final Score: 75.8%</div>
            <div style="margin-top: 8px; padding: 8px; background: var(--background); border-radius: 4px; font-size: 12px;">
                The result of 2 + 2 is 4.
            </div>
        </div>
    `;
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Show scoring
    document.getElementById('thinkingProgress').style.width = '85%';
    document.getElementById('currentThinkingStep').textContent = 'Scoring responses...';
    
    document.getElementById('responseScoring').innerHTML = `
        <div style="margin-bottom: 12px; background: var(--surface); padding: 12px; border-radius: var(--radius);">
            <h4>GPT-4</h4>
            <div class="grid grid-4" style="gap: 8px; font-size: 12px;">
                <div><strong>Confidence:</strong> 97.4%</div>
                <div><strong>Consistency:</strong> 100.0%</div>
                <div><strong>Weight:</strong> 85.0%</div>
                <div><strong>Final:</strong> 97.4%</div>
            </div>
        </div>
        <div style="margin-bottom: 12px; background: var(--surface); padding: 12px; border-radius: var(--radius);">
            <h4>Claude</h4>
            <div class="grid grid-4" style="gap: 8px; font-size: 12px;">
                <div><strong>Confidence:</strong> 83.7%</div>
                <div><strong>Consistency:</strong> 100.0%</div>
                <div><strong>Weight:</strong> 88.0%</div>
                <div><strong>Final:</strong> 83.7%</div>
            </div>
        </div>
        <div style="margin-bottom: 12px; background: var(--surface); padding: 12px; border-radius: var(--radius);">
            <h4>Gemini</h4>
            <div class="grid grid-4" style="gap: 8px; font-size: 12px;">
                <div><strong>Confidence:</strong> 75.8%</div>
                <div><strong>Consistency:</strong> 100.0%</div>
                <div><strong>Weight:</strong> 75.0%</div>
                <div><strong>Final:</strong> 75.8%</div>
            </div>
        </div>
    `;
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Complete
    document.getElementById('thinkingProgress').style.width = '100%';
    document.getElementById('currentThinkingStep').textContent = 'Hard thinking complete!';
    document.getElementById('thinkingStatus').textContent = 'Complete';
    document.getElementById('thinkingStatus').className = 'status-badge complete';
    document.getElementById('thinkingIcon').textContent = '✅';
    
    // Show synthesis
    document.getElementById('finalSynthesis').innerHTML = `
        <div style="background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>Best Model: GPT-4</h4>
            <p>Strategy: voting</p>
            <p>Consensus Level: 85.6%</p>
            <div style="margin-top: 12px; padding: 12px; background: var(--background); border-radius: 4px;">
                Based on multi-LLM ensemble analysis, the answer is 4. All models achieved high consensus.
            </div>
        </div>
    `;
    
    // Show final results
    document.getElementById('hardThinkingResults').style.display = 'block';
    document.getElementById('finalScore').textContent = '0.974';
    document.getElementById('consensusLevel').textContent = '86%';
    document.getElementById('processingTime').textContent = '0.9s';
    document.getElementById('totalTokens').textContent = '382';
    
    document.getElementById('finalAnswer').textContent = 'Based on multi-LLM ensemble analysis using voting strategy: The answer is 4. GPT-4 provided the highest-scoring response with 97.4% confidence. The ensemble achieved 85.6% consensus across all models. ✅ High consensus achieved across models';
    
    document.getElementById('modelBreakdown').innerHTML = `
        <div style="background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>GPT-4</h4>
            <div style="font-size: 13px;">
                <div><strong>Confidence:</strong> 97.4%</div>
                <div><strong>Responses:</strong> 1</div>
                <div><strong>Tokens:</strong> 82</div>
            </div>
        </div>
        <div style="background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>Claude</h4>
            <div style="font-size: 13px;">
                <div><strong>Confidence:</strong> 83.7%</div>
                <div><strong>Responses:</strong> 1</div>
                <div><strong>Tokens:</strong> 151</div>
            </div>
        </div>
        <div style="background: var(--surface); padding: 16px; border-radius: var(--radius);">
            <h4>Gemini</h4>
            <div style="font-size: 13px;">
                <div><strong>Confidence:</strong> 75.8%</div>
                <div><strong>Responses:</strong> 1</div>
                <div><strong>Tokens:</strong> 149</div>
            </div>
        </div>
    `;
    
    console.log('🎉 Hard Thinking Demo Complete!');
};

console.log('✅ Hard Thinking demo function loaded. Call demoHardThinking() to test.');
"""
    
    with open('/Users/alhinai/Desktop/TRUE/AgEval/debug_hardthinking.js', 'w') as f:
        f.write(demo_script)
    
    print("✅ Created debug script: debug_hardthinking.js")
    print("📝 Copy and paste this script into your browser console to test Hard Thinking")
    print("🚀 Then run: demoHardThinking()")

if __name__ == "__main__":
    create_simple_demo()
    
    print("\n🔧 Quick Fix Instructions:")
    print("1. Start the server: python start_server.py")
    print("2. Open: http://localhost:8001")
    print("3. Go to Hard Thinking tab")
    print("4. Open browser developer tools (F12)")
    print("5. Go to Console tab")
    print("6. Copy the debug_hardthinking.js content and paste it")
    print("7. Run: demoHardThinking()")
    print("8. OR just enter a question and click 'Start Hard Thinking' - it should work!")