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
        const data = await response.json();
        if (data.prediction) {
            let html = `<b>Result:</b> ${data.prediction.toUpperCase()}<br><b>Confidence:</b> ${(data.confidence*100).toFixed(2)}%`;
            if (data.keywords && data.keywords.length > 0) {
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