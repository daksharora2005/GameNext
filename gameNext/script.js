document.addEventListener("DOMContentLoaded", function () {
    console.log("Script Loaded");

    let userResponses = {
        "Action": 0,
        "Sports": 0,
        "Sci-Fi": 0,
        "Horror": 0,
        "Fantasy": 0,
        "Open World": 0,
        "Strategy": 0,
        "Shooter": 0,
        "Historical": 0,
        "Casual": 0,
        "Adult": 0
    };

    let questionCategories = [
        "Action", "Sports", "Sci-Fi", "Horror", "Fantasy", 
        "Open World", "Strategy", "Casual", "Shooter", 
        "Historical", "Adult"
    ];

    document.querySelectorAll(".question button").forEach((button, index) => {
        button.addEventListener("click", function () {
            let questionIndex = Math.floor(index / 5); 
            let score = parseInt(this.textContent.trim());
            
            userResponses[questionCategories[questionIndex]] = score;
            console.log("Updated Responses:", userResponses);
        });
    });

    document.getElementById("filter-submit").addEventListener("click", function () {
        console.log("Sending Responses:", userResponses);

        fetch("http://127.0.0.1:8000/get_recommendations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(userResponses)
        })
        .then(response => response.json())
        .then(data => {
            console.log("Server Response:", data);
            window.location.href = "index2.html";
        })
        .catch(error => console.error("Error:", error));
    });
});
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".question").forEach(question => {
        question.querySelectorAll("button").forEach(button => {
            button.addEventListener("click", function () {
                question.querySelectorAll("button").forEach(btn => btn.classList.remove("selected"));

                this.classList.add("selected");
            });
        });
    });
});
