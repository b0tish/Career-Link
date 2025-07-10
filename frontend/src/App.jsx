"use client"
import React, { useState } from "react"
import "./App.css" // <-- Import external CSS

export default function HomePage() {
    const [showAbout, setShowAbout] = useState(false)
    const [uploadedFile, setUploadedFile] = useState(null)

    const handleFileUpload = (event) => {
        const file = event.target.files?.[0]
        if (file) {
            setUploadedFile(file)
        }
    }

    const handleAboutToggle = () => {
        setShowAbout(!showAbout)
    }

    return (
        <div className="homepage-container">
            {/* Navigation */}
            <nav className="navbar">
                <div className="nav-inner">
                    <div className="logo">CareerLink</div>
                    <div className="nav-links">
                        <button onClick={() => setShowAbout(false)}>Home</button>
                        <button onClick={handleAboutToggle}>About</button>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="main-content">
                {/* Conditional rendering */}
                {!showAbout ? (
                    <>
                        {/* Hero (Welcome) */}
                        <section className="hero-section">
                            <h2>Welcome to CareerLink</h2>
                            <p>
                                Your gateway to career opportunities. Connect, grow, and succeed with our comprehensive career platform.
                            </p>
                            <div className="hero-buttons">
                                <button className="btn-primary">Get Started</button>
                                <button className="btn-outline">Learn More</button>
                            </div>
                        </section>

                        {/* Upload Section */}
                        <section className="upload-section">
                            <h2>Upload Your Documents</h2>
                            <p>Upload your resume, cover letter, or career documents to get started.</p>

                            <div className="upload-box">
                                <input type="file" id="file-upload" className="hidden" onChange={handleFileUpload} />
                                <label htmlFor="file-upload" className="upload-label">
                                    📄 Click to upload or drag and drop
                                    <br />
                                    <small>PDF, DOC, DOCX, TXT (max 10MB)</small>
                                </label>
                            </div>

                            {uploadedFile && (
                                <div className="upload-result">
                                    ✓ File uploaded successfully: {uploadedFile.name}
                                    <br />
                                    Size: {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                                </div>
                            )}

                            <button className="btn-primary mt-4">Process Document</button>
                        </section>

                        {/* Features Section (Job Opportunities etc.) */}
                        <section className="features-section">
                            {[
                                { icon: "💼", title: "Job Opportunities", desc: "Discover thousands of job openings from top companies." },
                                { icon: "🚀", title: "Career Growth", desc: "Tools to accelerate your professional development." },
                                { icon: "🤝", title: "Networking", desc: "Connect with industry professionals worldwide." },
                            ].map((item, i) => (
                                <div className="feature-card" key={i}>
                                    <div className="feature-icon">{item.icon}</div>
                                    <h3>{item.title}</h3>
                                    <p>{item.desc}</p>
                                </div>
                            ))}
                        </section>
                    </>
                ) : (
                    // About Section
                    <section className="about-section">
                        <h2>About CareerLink</h2>
                        <div className="about-card">
                            <p>
                                CareerLink is a career platform designed to bridge the gap between talented professionals and exciting
                                job opportunities. We provide tools and support for job seekers at every stage.
                            </p>
                            <p>
                                From resume building to professional networking, CareerLink empowers users to manage their career
                                journey confidently.
                            </p>
                            <div className="about-mission">
                                <div>
                                    <h3>Our Mission</h3>
                                    <p>To connect talent with opportunity globally.</p>
                                </div>
                                <div>
                                    <h3>Our Vision</h3>
                                    <p>Creating a future where everyone has access to career opportunities.</p>
                                </div>
                            </div>
                        </div>
                    </section>
                )}
            </main>

            {/* Footer */}
            <footer className="footer">
                <p>© 2024 CareerLink. All rights reserved. Empowering careers worldwide.</p>
            </footer>
        </div>
    )
}