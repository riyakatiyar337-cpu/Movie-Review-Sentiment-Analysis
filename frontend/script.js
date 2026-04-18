async function predictSentiment() {
    const text = document.getElementById("review").value;
    const model = document.getElementById("model").value;

    if (!text) {
        alert("Please enter a review!");
        return;
    }

    if (!model) {
        alert("Please select a model!");
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                model_name: model
            })
        });

        const data = await response.json();

        console.log("API Response:", data);

        if (data.error) {
            document.getElementById("result").innerHTML =
                "❌ Error: " + data.error;
            return;
        }

        document.getElementById("result").innerHTML = `
            <b>Model:</b> ${data.model.toUpperCase()} <br>
            <b>Sentiment:</b> ${data.sentiment} <br>
            <b>Confidence:</b> ${data.confidence ? data.confidence.toFixed(4) : "N/A"}
        `;

    } catch (error) {
        console.error("Fetch error:", error);
        document.getElementById("result").innerHTML =
            "❌ Failed to connect to API";
    }
}



async function compareModels() {
    const text = document.getElementById("review").value;

    if (!text) {
        alert("Please enter a review!");
        return;
    }

    const models = ["svm", "logreg", "nb", "rf", "lstm", "bilstm"];

    let resultsHTML = `
        <table border="1" style="width:100%; color:white; margin-top:20px;">
            <tr>
                <th>Model</th>
                <th>Sentiment</th>
                <th>Confidence</th>
            </tr>
    `;

    for (let model of models) {
        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    text: text,
                    model_name: model
                })
            });

            const data = await response.json();

            if (data.error) {
                resultsHTML += `
                    <tr>
                        <td>${model.toUpperCase()}</td>
                        <td colspan="2">Error</td>
                    </tr>
                `;
            } else {
                resultsHTML += `
                    <tr>
                        <td>${model.toUpperCase()}</td>
                        <td>${data.sentiment}</td>
                        <td>${data.confidence ? data.confidence.toFixed(4) : "N/A"}</td>
                    </tr>
                `;
            }

        } catch (error) {
            resultsHTML += `
                <tr>
                    <td>${model.toUpperCase()}</td>
                    <td colspan="2">Failed</td>
                </tr>
            `;
        }
    }

    resultsHTML += `</table>`;

    document.getElementById("comparison").innerHTML = resultsHTML;
}

async function loadLeaderboard() {
    try {
        const response = await fetch("http://127.0.0.1:8000/leaderboard");
        const data = await response.json();

        if (data.error) {
            document.getElementById("leaderboard").innerHTML =
                "❌ " + data.error;
            return;
        }

        let html = `
            <h3>🏆 Accuracy Leaderboard</h3>
            <table border="1" style="width:100%; color:white;">
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Accuracy</th>
                </tr>
        `;

        let rank = 1;

        for (let model in data) {
            html += `
                <tr>
                    <td>${rank}</td>
                    <td>${model.toUpperCase()}</td>
                    <td>${(data[model] * 100).toFixed(2)}%</td>
                </tr>
            `;
            rank++;
        }

        html += `</table>`;

        document.getElementById("leaderboard").innerHTML = html;

    } catch (error) {
        document.getElementById("leaderboard").innerHTML =
            "❌ Failed to load leaderboard";
    }
}