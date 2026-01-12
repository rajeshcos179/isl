const API_BASE = "http://localhost:8000";

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/video-to-text`, {
    method: "POST",
    body: formData,
  });

  return res.json();
}

export async function textToSign(text) {
  const res = await fetch(`${API_BASE}/text-to-sign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  return res.json();
}
