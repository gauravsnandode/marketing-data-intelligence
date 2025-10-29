import requests
import json

# The base URL of your running FastAPI application
BASE_URL = "http://127.0.0.1:8000"

# --- Test Cases for /predict_discount ---
predict_test_case_1 = {
    "category": 10,          # Example category ID
    "actual_price": 2999.0,  # A mid-range price
    "rating": 4.5,           # A high rating
    "rating_count": 12500.0  # A high number of ratings
}

predict_test_case_2 = {
    "category": 5,           # A different category
    "actual_price": 499.0,   # A lower price
    "rating": 3.8,           # A moderate rating
    "rating_count": 800.0    # Fewer ratings
}

# --- Test Cases for /answer_question ---
answer_test_case_1 = {
    "query": "are there any good wireless earbuds?",
    "top_k": 2
}

answer_test_case_2 = {
    "query": "which monitor is good for gaming"
}

def test_endpoint(endpoint: str, payload: dict, test_name: str):
    """
    Sends a POST request to a specified API endpoint and prints the result.
    """
    api_url = f"{BASE_URL}{endpoint}"
    print(f"--- Running {test_name} ---")
    print(f"Hitting endpoint: {api_url}")
    try:
        # Send the POST request with the JSON data
        response = requests.post(api_url, json=payload)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Print the successful response from the server
        print("Request successful!")
        print(f"Input: {json.dumps(payload)}")
        # Use json.dumps for pretty printing the response
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}\n")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error: Could not connect to the API at {api_url}.")
        print("Please ensure the FastAPI server is running.\n")
    except Exception as err:
        print(f"An unexpected error occurred: {err}\n")

if __name__ == "__main__":
    # Test the discount prediction endpoint
    test_endpoint("/predict_discount", predict_test_case_1, "Predict Discount Test 1")
    test_endpoint("/predict_discount", predict_test_case_2, "Predict Discount Test 2")

    # Test the question answering endpoint
    test_endpoint("/answer_question", answer_test_case_1, "Answer Question Test 1")
    test_endpoint("/answer_question", answer_test_case_2, "Answer Question Test 2")
