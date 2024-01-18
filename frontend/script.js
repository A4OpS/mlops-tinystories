document.getElementById('queryForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevents the form from submitting the traditional way
    queryAPI();
});

function queryAPI() {
    let input = document.getElementById('userInput').value;
    let url = `/generate/model?input=${encodeURIComponent(input)}`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerText = data;
        })
        .catch(error => console.error('Error:', error));
}