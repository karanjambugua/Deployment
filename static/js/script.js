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

function refreshPage() {
    location.reload();  // This will reload the current page
}
// Function to update the circular progress meter
function updateProgressCircle(progress) {
    const circleProgress = document.querySelector('.circle-progress');
    const progressText = document.getElementById('progress-text');

    // Calculate stroke-dashoffset based on progress (0-100%)
    const offset = 440 - (progress / 100) * 440;  // 440 is the circumference
    circleProgress.style.strokeDashoffset = offset;

    // Update the text in the center of the circle
    progressText.innerText = `${Math.round(progress)}%`;
}

async function analyzeTweet() {
    const tweet = document.getElementById('tweet').value;
    const result = document.getElementById('result');
    const progressContainer = document.getElementById('progress-container');
    
    if (!tweet) {
        alert("Please enter a tweet.");
        return;
    }

    // Show the progress container and start animating the liquid
    progressContainer.style.display = 'flex';  // Show the progress meter
    updateProgressCircle(0);  // Initialize progress to 0%

    // Animate fluid filling in stages (0 -> 50 -> 80 -> 100)
    let progress = 0;
    const interval = setInterval(() => {
        if (progress < 50) {
            progress += 10;
        } else if (progress < 80) {
            progress += 5;
        } else if (progress < 100) {
            progress += 5;
        }
        updateProgressCircle(progress); // Update progress
    }, 1000);

    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ tweet: tweet })
    });

    if (!response.ok) {
        alert("Error analyzing tweet.");
        clearInterval(interval);
        return;
    }

    const data = await response.json();

    // Display the sentiment result after the analysis
    setTimeout(() => {
        result.innerText = `Sentiment: ${data.sentiment}`;
    }, 5000);  // Show sentiment after animation is complete
}
