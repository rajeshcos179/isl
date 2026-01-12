import { useState } from "react";
import { uploadVideo, textToSign } from "./api";
import "./index.css";

function App() {
  const [video, setVideo] = useState(null);
  const [englishText, setEnglishText] = useState("");
  const [inputText, setInputText] = useState("");
  const [animationUrl, setAnimationUrl] = useState("");
  const [showModal, setShowModal] = useState(false);

  const handleVideoUpload = async () => {
    if (!video) return alert("Select a video!");

    setEnglishText("How are you")
    // const result = await uploadVideo(video);
    // setEnglishText(result.english_text);
  };

  const handleTextSubmit = async () => {
    if (!inputText) return alert("Enter text!");

    // const result = await textToSign(inputText);
    // setAnimationUrl(result.animation_url);

    // Show modal container
    setShowModal(true);
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
                />
                <button onClick={handleVideoUpload}>
                  Translate Video
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
                />

                <button onClick={handleTextSubmit}>
                  Generate Sign Animation
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
