document.getElementById('newsForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    console.log('Form submitted!'); // Debug line
    const title = document.getElementById('title').value;
    const body = document.getElementById('body').value;
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Checking...';
    try {
        const response = await fetch('api/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title, body })
        });
        if (response.status === 401) {
            window.location.href = '/accounts/login/';
            return;
        }
        const data = await response.json();
        if (data.prediction) {
            const confidencePercent = data.confidence * 100;
            const certainty = confidencePercent > 70 ? 'Probably' : 'Might be';
            let html = `<b>Result:</b> ${certainty} ${data.prediction.toUpperCase()}<br><b>Confidence:</b> ${confidencePercent.toFixed(2)}%`;
            if (data.similarity) {
                html += `<br><b>Title-Body Similarity:</b> ${data.similarity}`;
            }
            if (confidencePercent > 70 && data.keywords && data.keywords.length > 0) {
                html += `<br><b>Top Contributing words for result:</b> <span style='color:#d9534f'>${data.keywords.join(', ')}</span>`;
            }
            resultDiv.innerHTML = html;
        } else if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
        } else {
            resultDiv.textContent = 'Unexpected response.';
        }
    } catch (err) {
        resultDiv.textContent = 'Error: ' + err;
    }
}); 