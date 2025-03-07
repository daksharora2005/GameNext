const games = Array.from({ length: 100 }, (_, i) => ({
    title: `Game ${i + 1}`,
    // image: `https://via.placeholder.com/200x250?text=Game+${i + 1}`,
    rating: ``,
    date_release: ``
}));

function displayGames() {
    const container = document.getElementById("games-container");
    container.innerHTML = "";
    games.forEach(game => {
        container.innerHTML += `
            <div class="game-card">
                <h3>${game.title}</h3>
                <p>Release date: ${game.date_release}</p>
                <p>Rating: ${game.rating}</p>
            </div>
        `;
    });
}

function sortGames() {
    const sortBy = document.getElementById("sort").value;
    if (sortBy === "rating") {
        games.sort((a, b) => b.rating - a.rating);
    } else if (sortBy === "date_release") {
        games.sort((a, b) => new Date(b.date_release) - new Date(a.date_release));
    } else {
        games.sort((a, b) => a.title.localeCompare(b.title));
    }
    displayGames();
}

window.onload = () => {
    sortGames();
};

// --- Additional code to integrate FastAPI recommendations ---
const userResponses = {};
async function fetchRecommendations() {
    try {
        const response = await fetch("http://127.0.0.1:8000/get_recommendations", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ responses: userResponses })
        })
        .then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error("Error:", error));
        const data = await response.json();
        console.log(response);
        if (data && data.recommendations) {
            // Clear the current dummy games array
            games.length = 0;
            // Map the backend recommendation data to match your expected game format
            data.recommendations.forEach(rec => {
                games.push({
                    title: rec.title,
                    // image: rec.image ? rec.image : `https://via.placeholder.com/200x250?text=${encodeURIComponent(rec.title)}`,
                    rating: rec.rating,
                    date_release: rec.date_release_release ? rec.date_release_release : (rec.date_release || "Unknown")
                });
            });
            // Re-run the sorting and display functions to update the page
            sortGames();
        }
    } catch (error) {
        console.error("Error fetching recommendations:", error);
    }
}

// Add an event listener so that recommendations are fetched when the page loads
window.addEventListener("load", fetchRecommendations);