"use client";
import { useState, useEffect, useRef } from "react";

export default function Home() {
  const inputRef = useRef(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [retrainDate, setRetrainDate] = useState(null);

  function formatDate(date) {
    return date.toLocaleString("en-US", {
      month: "short",
      day: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  }

  function handleRetrain() {
    setRetrainDate(new Date());
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await fetch("https://mlop-api.onrender.com/api/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "Failed to get prediction." });
    }
    setLoading(false);
  };

  return (
    <section className="home">
      <nav className="navigation">
        <h1>classtra ai</h1>
        <div className="navigation-links">
          <a
            className="navigation-links-cta"
            href="https://github.com/jmugisha1/MLOP-Traffic-Image-Classification"
          >
            code repo
          </a>
          <button className="navigation-links-cta" onClick={handleRetrain}>
            retrain model
          </button>
        </div>
      </nav>
      <main className="home-main">
        <h1 className="home-heading">
          Hello i’m classtra an ml model, i was created to do
          <br />
          classification on traffic vehicle images
        </h1>

        <div className="home-prompt">
          {/*  */}
          <div className="prompt-classes">
            <p className="prompt-classes-p">
              image are classified with in 4 classes
            </p>
            {/*  */}
            <div className="prompt-classes-icons-wrapper">
              <div className="prompt-classes-icons">
                <img
                  className="prompt-classes-icons-i"
                  src="/icons/car-profile.svg"
                ></img>
                <p className="prompt-classes-icons-p">car</p>
              </div>
              <div className="prompt-classes-icons">
                <img
                  className="prompt-classes-icons-i"
                  src="/icons/motorcycle.svg"
                ></img>
                <p className="prompt-classes-icons-p">motorcycle</p>
              </div>
              <div className="prompt-classes-icons">
                <img
                  className="prompt-classes-icons-i"
                  src="/icons/truck.svg"
                ></img>
                <p className="prompt-classes-icons-p">truck</p>
              </div>
              <div className="prompt-classes-icons">
                <img
                  className="prompt-classes-icons-i"
                  src="/icons/van.svg"
                ></img>
                <p className="prompt-classes-icons-p">bus</p>
              </div>
            </div>
          </div>

          {/*  */}
          <form onSubmit={handleSubmit} className="prompt-upload">
            <div className="prompt-upload-input">
              <img
                src="/icons/paperclip.svg"
                className="prompt-upload-input-i"
                alt="Attach"
                onClick={() => inputRef.current && inputRef.current.click()}
              />
              <label htmlFor="attach">attach</label>
              <input
                ref={inputRef}
                id="attach"
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                required
                onChange={(e) => setImage(e.target.files[0])}
              />
            </div>
            <button
              type="submit"
              className="prompt-upload-submit"
              disabled={loading}
            >
              <img
                src="/icons/arrow-up.svg"
                className="prompt-upload-submit-i"
              />
            </button>
          </form>
        </div>
        <p className="prompt-upload-preview">
          {image ? `Selected image is: ${image.name}` : "No image selected"}
        </p>
        <div className="prompt-result">
          {loading && <p>Loading...</p>}
          {result && result.error && (
            <p className="prompt-result-p">{result.error}</p>
          )}
          {result && result.predicted_class && (
            <>
              <p className="prompt-result-p">image is classified as</p>
              <p className="prompt-result-r">{result.predicted_class}</p>
              <p className="prompt-result-p">with an accuracy</p>
              <p className="prompt-result-r">
                {(result.confidence * 100).toFixed(2)}%
              </p>
            </>
          )}
        </div>
      </main>
      <footer className="home-footer">
        <p className="home-footer-p">
          {retrainDate
            ? `Model retrained — ${formatDate(retrainDate)}`
            : "Model not retrained yet"}
        </p>
      </footer>
    </section>
  );
}
