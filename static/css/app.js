document.getElementById('newsForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Check if user is authenticated - only authenticated users should be able to submit
    const submitButton = document.querySelector('#newsForm button[type="submit"]');
    if (!submitButton || submitButton.hasAttribute('disabled')) {
        window.location.href = '/accounts/login/';
        return;
    }
    
    console.log('Form submitted!'); // Debug line
    
    // Restore original inputs if they were hidden by highlighting
    const titleInput = document.getElementById('title');
    const bodyTextarea = document.getElementById('body');
    const titleHighlighted = document.getElementById('title-highlighted');
    const bodyHighlighted = document.getElementById('body-highlighted');
    
    if (titleHighlighted && titleHighlighted.style.display !== 'none') {
        titleInput.style.display = 'block';
        titleHighlighted.style.display = 'none';
    }
    if (bodyHighlighted && bodyHighlighted.style.display !== 'none') {
        bodyTextarea.style.display = 'block';
        bodyHighlighted.style.display = 'none';
    }
    
    const title = titleInput.value;
    const body = bodyTextarea.value;
    const resultDiv = document.getElementById('result');
    
    // Show loading state
    resultDiv.classList.remove('hidden');
    resultDiv.className = 'mt-6 p-5 bg-black rounded-lg border border-gray-800 min-h-[80px] loading';
    resultDiv.innerHTML = '<div class="flex items-center justify-center"><div class="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-400"></div><span class="ml-3 text-gray-300 text-sm">Analyzing...</span></div>';
    
    // Get CSRF token from form or cookies
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || getCookie('csrftoken');
    
    try {
        const response = await fetch('api/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken || ''
            },
            body: JSON.stringify({ title, body })
        });
        
        if (response.status === 401) {
            const data = await response.json();
            resultDiv.className = 'mt-6 p-5 bg-red-900/30 border border-red-700 rounded-lg min-h-[80px]';
            resultDiv.innerHTML = `
                <div class="text-red-200 text-center">
                    <div class="text-base font-medium mb-2">Sign in required</div>
                    <div class="text-sm mb-3 text-red-300">${data.message || 'Please log in to check articles.'}</div>
                    <a href="/accounts/login/" class="inline-block bg-red-700 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-lg transition-colors text-sm">
                        Log in
                    </a>
                </div>
            `;
            return;
        }
        
        const data = await response.json();
        
        // Debug: Log the full response
        console.log('Full API response:', data);
        
        if (data.prediction) {
            const confidencePercent = data.confidence * 100;
            const certainty = confidencePercent > 70 ? 'Likely' : 'Possibly';
            const isFake = data.prediction.toLowerCase() === 'fake';
            const resultColor = isFake ? 'text-red-300' : 'text-green-300';
            const resultBg = isFake ? 'bg-red-900/30 border-red-700' : 'bg-green-900/30 border-green-700';
            
            resultDiv.className = `mt-6 p-5 ${resultBg} border rounded-lg min-h-[80px]`;
            
            // Get top_words (support both top_words and keywords for backwards compatibility)
            const topWords = data.top_words || data.keywords || [];
            
            console.log('Top words from response:', topWords);
            console.log('Top words type:', typeof topWords);
            console.log('Top words length:', topWords ? topWords.length : 'undefined');
            
            let html = `
                <div>
                    <div class="text-lg font-semibold mb-2 ${resultColor}">${certainty} ${data.prediction.charAt(0).toUpperCase() + data.prediction.slice(1)}</div>
                    <div class="text-sm text-gray-400 mb-3">Confidence: ${confidencePercent.toFixed(1)}%</div>
            `;
            
            if (data.similarity) {
                html += `<div class="text-sm text-gray-400 mb-2">Title-content match: ${data.similarity}</div>`;
            }
            
            if (topWords && topWords.length > 0) {
                html += `<div class="text-sm text-gray-400 mt-3 pt-3 border-t border-gray-700"><span class="text-gray-300">Notable words:</span> <span class="text-gray-500">${topWords.join(', ')}</span></div>`;
            } else {
                html += `<div class="text-sm text-gray-400 mt-3 pt-3 border-t border-gray-700"><span class="text-gray-300">Notable words:</span> <span class="text-gray-500">None found</span></div>`;
            }
            
            html += '</div>';
            resultDiv.innerHTML = html;
            
            // Highlight top words in the article text
            console.log('About to highlight. Top words:', topWords, 'Length:', topWords ? topWords.length : 0);
            if (topWords && Array.isArray(topWords) && topWords.length > 0) {
                console.log('Calling highlightTopWords with:', topWords.length, 'words');
                console.log('Title:', title);
                console.log('Body length:', body ? body.length : 0);
                try {
                    highlightTopWords(title, body, topWords);
                    console.log('highlightTopWords called successfully');
                } catch (error) {
                    console.error('Error in highlightTopWords:', error);
                }
            } else {
                console.log('Skipping highlight - no top words or empty array');
                console.log('topWords value:', topWords);
                console.log('Is array?', Array.isArray(topWords));
            }
        } else if (data.error) {
            resultDiv.className = 'mt-6 p-5 bg-red-900/30 border border-red-700 rounded-lg min-h-[80px]';
            resultDiv.innerHTML = `
                <div class="text-red-200 text-center">
                    <div class="text-sm font-medium mb-1">Error</div>
                    <div class="text-xs text-red-300">${data.error}</div>
                </div>
            `;
        } else {
            resultDiv.className = 'mt-6 p-5 bg-yellow-900/30 border border-yellow-700 rounded-lg min-h-[80px]';
            resultDiv.innerHTML = `
                <div class="text-yellow-200 text-center">
                    <div class="text-sm font-medium mb-1">Unexpected response</div>
                    <div class="text-xs text-yellow-300">Please try again later.</div>
                </div>
            `;
        }
    } catch (err) {
        resultDiv.className = 'mt-6 p-5 bg-red-900/30 border border-red-700 rounded-lg min-h-[80px]';
        resultDiv.innerHTML = `
            <div class="text-red-200 text-center">
                <div class="text-sm font-medium mb-1">Connection error</div>
                <div class="text-xs text-red-300">Check your connection and try again.</div>
            </div>
        `;
        console.error('Error:', err);
    }
});

// Function to highlight top words in the article text
function highlightTopWords(title, body, topWords) {
    console.log('=== highlightTopWords called ===');
    console.log('Title:', title);
    console.log('Body:', body);
    console.log('Top words:', topWords);
    
    if (!topWords || !Array.isArray(topWords) || topWords.length === 0) {
        console.log('Returning early - no top words');
        return;
    }
    
    console.log('Starting highlight process with', topWords.length, 'words');
    
    // Clean up any existing highlighted displays
    const existingTitleHighlight = document.getElementById('title-highlighted');
    const existingBodyHighlight = document.getElementById('body-highlighted');
    const existingNote = document.getElementById('highlight-note');
    
    if (existingTitleHighlight) {
        console.log('Removing existing title highlight');
        existingTitleHighlight.remove();
    }
    if (existingBodyHighlight) {
        console.log('Removing existing body highlight');
        existingBodyHighlight.remove();
    }
    
    // Get original inputs
    const titleInput = document.getElementById('title');
    const bodyTextarea = document.getElementById('body');
    
    console.log('Title input found:', !!titleInput);
    console.log('Body textarea found:', !!bodyTextarea);
    
    if (!titleInput || !bodyTextarea) {
        console.error('Inputs not found - titleInput:', !!titleInput, 'bodyTextarea:', !!bodyTextarea);
        return;
    }
    
    // Create a Set for faster lookup (case-insensitive)
    const topWordsSet = new Set(topWords.map(word => word.toLowerCase()));
    
    // Function to highlight words in text
    function highlightText(text, wordsSet) {
        if (!text) return '';
        
        // Escape special regex characters in words
        const escapedWords = Array.from(wordsSet).map(word => 
            word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        );
        
        // Create regex pattern to match whole words only (case-insensitive)
        const pattern = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'gi');
        
        // Replace matches with highlighted version
        return text.replace(pattern, (match) => {
            return `<mark class="bg-yellow-500/30 text-yellow-200 font-semibold px-1 rounded">${match}</mark>`;
        });
    }
    
    // Highlight title
    if (title && title.trim()) {
        const titleContainer = titleInput.parentElement;
        
        // Create highlighted title display
        const titleDisplay = document.createElement('div');
        titleDisplay.className = 'w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white text-base min-h-[48px]';
        titleDisplay.innerHTML = highlightText(title, topWordsSet);
        titleDisplay.id = 'title-highlighted';
        titleDisplay.style.cssText = 'display: block !important; width: 100%;';
        
        // Insert highlighted display right before the input, then hide the input
        titleContainer.insertBefore(titleDisplay, titleInput);
        titleInput.style.cssText = 'display: none !important;';
        
        // Add click handler to switch back to edit mode
        titleDisplay.addEventListener('dblclick', function() {
            titleInput.style.cssText = 'display: block !important;';
            titleDisplay.style.cssText = 'display: none !important;';
        });
        
        console.log('Title highlighted, input hidden, display inserted');
        console.log('Title container:', titleContainer);
        console.log('Title display element:', titleDisplay);
        console.log('Title display innerHTML length:', titleDisplay.innerHTML.length);
        console.log('Title display computed style:', window.getComputedStyle(titleDisplay).display);
        console.log('Title input computed style:', window.getComputedStyle(titleInput).display);
        
        // Verify element is in DOM
        setTimeout(() => {
            const checkDisplay = document.getElementById('title-highlighted');
            console.log('Verification - title-highlighted in DOM:', !!checkDisplay);
            if (checkDisplay) {
                console.log('Title display is visible:', checkDisplay.offsetParent !== null);
            }
        }, 100);
    } else {
        console.log('Title not highlighted - title is empty or null');
    }
    
    // Highlight body
    if (body && body.trim()) {
        const bodyContainer = bodyTextarea.parentElement;
        
        // Create highlighted body display
        const bodyDisplay = document.createElement('div');
        bodyDisplay.className = 'w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white text-base resize-none overflow-y-auto whitespace-pre-wrap';
        bodyDisplay.style.cssText = 'min-height: 250px; max-height: 400px; display: block !important; width: 100%;';
        bodyDisplay.innerHTML = highlightText(body, topWordsSet).replace(/\n/g, '<br>');
        bodyDisplay.id = 'body-highlighted';
        
        // Insert highlighted display right before the textarea, then hide the textarea
        bodyContainer.insertBefore(bodyDisplay, bodyTextarea);
        bodyTextarea.style.cssText = 'display: none !important;';
        
        // Add click handler to switch back to edit mode
        bodyDisplay.addEventListener('dblclick', function() {
            bodyTextarea.style.cssText = 'display: block !important;';
            bodyDisplay.style.cssText = 'display: none !important;';
        });
        
        console.log('Body highlighted, textarea hidden, display inserted');
        console.log('Body container:', bodyContainer);
        console.log('Body display element:', bodyDisplay);
        console.log('Body display innerHTML length:', bodyDisplay.innerHTML.length);
        console.log('Body display computed style:', window.getComputedStyle(bodyDisplay).display);
        console.log('Body textarea computed style:', window.getComputedStyle(bodyTextarea).display);
        
        // Verify element is in DOM
        setTimeout(() => {
            const checkDisplay = document.getElementById('body-highlighted');
            console.log('Verification - body-highlighted in DOM:', !!checkDisplay);
            if (checkDisplay) {
                console.log('Body display is visible:', checkDisplay.offsetParent !== null);
            }
        }, 100);
    } else {
        console.log('Body not highlighted - body is empty or null');
    }
    
    // Add or update note to inform users they can double-click to edit
    if (!existingNote) {
        const noteDiv = document.createElement('div');
        noteDiv.className = 'text-xs text-gray-500 mt-2 mb-2';
        noteDiv.innerHTML = '💡 Double-click on highlighted text to edit';
        noteDiv.id = 'highlight-note';
        
        // Insert note before the result div
        const resultDiv = document.getElementById('result');
        if (resultDiv) {
            resultDiv.parentElement.insertBefore(noteDiv, resultDiv);
            console.log('Added highlight note');
        } else {
            console.log('Result div not found, cannot add note');
        }
    }
    
    console.log('=== highlightTopWords completed ===');
} 