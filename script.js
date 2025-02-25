document.addEventListener("DOMContentLoaded", function () {
    let ctx = document.getElementById("accuracyChart").getContext("2d");
    let accuracyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Model Accuracy",
                data: [],
                borderColor: "#007bff",
                backgroundColor: "rgba(0, 123, 255, 0.2)",
                borderWidth: 2,
                pointRadius: 4,
                fill: true,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            elements: {
                line: {
                    tension: 0.3
                }
            },
            layout: {
                padding: 10
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: "Accuracy (%)"
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: "Test Runs"
                    }
                }
            },
            animation: false
        }
    });

    window.updateChart = function (accuracy) {
        if (isNaN(accuracy)) {
            console.error("Invalid accuracy value received:", accuracy);
            return;
        }
        let labels = accuracyChart.data.labels;
        let data = accuracyChart.data.datasets[0].data;

        if (labels.length > 20) {
            labels.shift();
            data.shift();
        }

        labels.push(labels.length + 1);
        data.push(accuracy);
        accuracyChart.update();
    };

    async function testAdaline() {
        const learningRate = document.getElementById('learning-rate').value;
        const epochs = document.getElementById('epochs').value;

        await fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ learningRate, epochs })
        });

        const response = await fetch('/test', { method: 'GET' });
        const result = await response.json();

        

        console.log("Received response from /test:", result);
        
        if (result.accuracy !== undefined) {
            document.getElementById('weightBox').innerHTML = `
                <p>Accuracy: ${result.accuracy.toFixed(2)}%</p>
                <p>Weights: ${Array.isArray(result.weights) ? result.weights.join(', ') : 'N/A'}</p>
                <div id="weights">${result.bias !== undefined ? result.bias.toFixed(4) : 'N/A'}</div>
            `;
            updateChart(result.accuracy);
        } else {
            console.error("Missing accuracy in response:", result);
        }
    }

    window.testAdaline = testAdaline;
});

