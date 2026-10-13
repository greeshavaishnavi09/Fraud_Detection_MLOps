

const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

export async function predictFraud(transaction) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(transaction),
  });

  const data = await response.json();

  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(", ")
      : detail || "Prediction request failed.";

    throw new Error(message);
  }

  return data;
}
