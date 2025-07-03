document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            displayError(result.detail || 'Analysis failed');
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

function displayResults(data) {
    let html = '<div class="container"><h2>📊 Analysis Results</h2>';
    
    // Chapter predictions
    html += '<h3>🎯 Chapter Predictions</h3>';
    data.chapter_predictions.forEach(pred => {
        const probPercent = (pred.probability * 100).toFixed(1);
        html += `<div class="chapter">
            <strong>${pred.chapter_name}</strong><br>
            <span class="score ${getScoreClass(pred.probability)}">${probPercent}%</span>
            <p><em>${pred.reasoning}</em></p>
        </div>`;
    });
    
    // Final recommendations
    if (data.final_codes && data.final_codes.length > 0) {
        html += '<h3>✅ Recommended ICD-10-CM Codes</h3>';
        data.final_codes.forEach(code => {
            const confPercent = (code.confidence_score * 100).toFixed(1);
            html += `<div class="code ${getConfidenceClass(code.confidence_score)}">
                <h4>${code.icd_code} - ${code.description}</h4>
                <span class="score ${getScoreClass(code.confidence_score)}">${confPercent}%</span>
                <p><strong>Reasoning:</strong> ${code.reasoning}</p>
            </div>`;
        });
        
        html += `<div class="result-box">
            <h4>💡 Overall Recommendation</h4>
            <p>${data.overall_recommendation}</p>
        </div>`;
    } else {
        html += '<div class="result-box"><p>❌ No high-confidence codes found. Please review the document or try with more detailed medical information.</p></div>';
    }
    
    html += '</div>';
    document.getElementById('results').innerHTML = html;
}

function displayError(message) {
    document.getElementById('results').innerHTML = 
        `<div class="container"><div class="result-box" style="border-left-color: #dc3545;">
            <h3>❌ Error</h3><p>${message}</p>
        </div></div>`;
}

function getScoreClass(score) {
    if (score >= 0.7) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
}

function getConfidenceClass(score) {
    if (score >= 0.7) return 'high-confidence';
    if (score >= 0.5) return 'medium-confidence';
    return 'low-confidence';
} 