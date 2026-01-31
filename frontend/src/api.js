const API_BASE = "http://localhost:8000";

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/video-to-text`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    const data = await res.json();
    return {
      text: data.text || data.gloss || "No text detected",
      gloss: data.gloss
    };
  } catch (error) {
    console.error("Error uploading video:", error);
    throw error;
  }
}

export async function textToSign(text) {
  console.log("Sending text to backend:", text);
  
  try {
    const res = await fetch(`${API_BASE}/text-to-pose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    const data = await res.json();
    console.log("Response from backend:", data);
    return data;
  } catch (error) {
    console.error("Error in textToSign:", error);
    throw error;
  }
}
