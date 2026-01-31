import { useState } from "react";
import { uploadVideo, textToSign } from "./api";
import "./index.css";

function App() {
  const [video, setVideo] = useState(null);
  const [englishText, setEnglishText] = useState("");
  const [inputText, setInputText] = useState("");
  const [animationUrl, setAnimationUrl] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleVideoUpload = async () => {
    if (!video) return alert("Select a video!");

    setLoading(true);
    try {
      const result = await uploadVideo(video);
      setEnglishText(result.text);
      console.log("Translation result:", result);
    } catch (error) {
      console.error("Error during translation:", error);
      alert("Error translating video. Please check the console.");
    } finally {
      setLoading(false);
    }
  };

  const handleTextSubmit = async () => {
    if (!inputText) return alert("Enter text!");

    try {
      setLoading(true);
      const result = await textToSign(inputText);
      
      // Console log the pose JSON data
      console.log("=== Pose Data JSON ===");
      console.log(JSON.stringify(result, null, 2));
      console.log("=== Pose Data Object ===");
      console.log(result);
      
      if (result.status === "success") {
        console.log("✅ Successfully generated poses for:", result.glosses);
        console.log("Pose data keys:", Object.keys(result.pose_data));
      }

      // Show modal container
      setShowModal(true);
    } catch (error) {
      console.error("❌ Error generating sign animation:", error);
      alert("Error generating animation. Please check the console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Blur wrapper */}
      <div className={`app-wrapper ${showModal ? "blurred" : ""}`}>
        <div className="container">

          <div className="header">
            <h1>🤟 Bidirectional ISL Translator</h1>
            <p>
              Translate Indian Sign Language videos into English and generate
              realistic ISL avatar animations using Dual-Learning Transformers.
            </p>
          </div>

          <div className="grid">

            {/* Video to Text */}
            <div className="card">
              <h2>🎥 ISL Video → English</h2>

              <div className="field">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setVideo(e.target.files[0])}
                  disabled={loading}
                />
                <button onClick={handleVideoUpload} disabled={loading}>
                  {loading ? "Translating..." : "Translate Video"}
                </button>
              </div>

              {englishText && (
                <div className="output">
                  <strong>Predicted Text</strong>
                  <p>{englishText}</p>
                </div>
              )}
            </div>

            {/* Text to Sign */}
            <div className="card">
              <h2>📝 English → ISL Avatar</h2>

              <div className="field">
                <input
                  type="text"
                  placeholder="Enter English sentence..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={loading}
                />

                <button onClick={handleTextSubmit} disabled={loading}>
                  {loading ? "Generating..." : "Generate Sign Animation"}
                </button>
              </div>
            </div>

          </div>

          <div className="footer">
            © 2026 • Bidirectional ISL Translation System
          </div>

        </div>
      </div>

      {/* Modal Overlay */}
      {showModal && (
        <div className="modal-overlay">
          <div className="modal-box">
            <button className="close-btn" onClick={() => setShowModal(false)}>
              ✕
            </button>

            <h2 className="text-center">🧍 ISL Avatar</h2>

            {/* Blank container */}
            <div className="avatar-placeholder">
              {/* Future 3D avatar / animation will go here */}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
