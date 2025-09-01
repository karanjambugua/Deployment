async function analyzeTweet() {
    const tweet = document.getElementById('tweet').value;
    const loader = document.getElementById('loader');
    const result = document.getElementById('result');

    if (!tweet) {
        alert("Please enter a tweet.");
        return;
    }

    // Show the loading spinner
    loader.style.display = 'inline-block';
    
    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ tweet: tweet })
    });

    if (!response.ok) {
        alert("Error analyzing tweet.");
        loader.style.display = 'none';
        return;
    }

    const data = await response.json();
    
    // Hide the loading spinner
    loader.style.display = 'none';
    
    // Display the sentiment result
    result.innerText = `Sentiment: ${data.sentiment}`;
}

