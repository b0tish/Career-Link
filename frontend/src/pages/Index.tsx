import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Navigation } from "@/components/Navigation";
import { UploadSection } from "@/components/UploadSection";
import { JobCard } from "@/components/JobCard";
import { Star, TrendingUp, ArrowRight } from "lucide-react";

const Index = () => {
	const [showAbout, setShowAbout] = useState(false);
	const [uploadedFile, setUploadedFile] = useState<File | null>(null);
	const [isProcessing, setIsProcessing] = useState(false);
	const [showJobs, setShowJobs] = useState(false);
	const [jobs, setJobs] = useState([]);

	const scrollToUpload = () => {
		const uploadSection = document.getElementById("upload-section");
		if (uploadSection) {
			uploadSection.scrollIntoView({ behavior: "smooth" });
		}
	};

	const handleFileUpload = (file: File) => {
		setUploadedFile(file);
	};

	const handleProcess = async () => {
		if (!uploadedFile) return;

		setIsProcessing(true);
		const formData = new FormData();
		formData.append("resume", uploadedFile);

		try {
			const response = await fetch("http://localhost:8000/api/upload/", {
				method: "POST",
				body: formData,
			});

			if (response.ok) {
				const data = await response.json();
				setJobs(data.top_jobs);
				setShowJobs(true);
				setTimeout(() => {
					const jobsSection = document.getElementById("jobs-section");
					if (jobsSection) {
						jobsSection.scrollIntoView({ behavior: "smooth" });
					}
				}, 100);
			} else {
				console.error("Error uploading file");
			}
		} catch (error) {
			console.error("Error:", error);
		} finally {
			setIsProcessing(false);
		}
	};

	const handleHomeClick = () => {
		setShowAbout(false);
		window.scrollTo({ top: 0, behavior: "smooth" });
	};

	const handleAboutClick = () => {
		setShowAbout(true);
	};

	return (
		<div className="min-h-screen bg-background">
			<Navigation
				showAbout={showAbout}
				onHomeClick={handleHomeClick}
				onAboutClick={handleAboutClick}
			/>

			{/* Main Content */}
			<main className="pt-20">
				{!showAbout ? (
					<>
						{/* Hero Section */}
						<section className="relative overflow-hidden py-32 px-6">
							<div className="absolute inset-0  opacity-10"></div>
							<div className="relative max-w-6xl mx-auto text-center">
								<h1 className="text-6xl md:text-7xl font-sans font-bold mb-8  text-primary leading-tight">
									Welcome to CareerLink
								</h1>
								<p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto leading-relaxed font-sans">
									Your gateway to career opportunities. Connect, grow, and
									succeed with our comprehensive career platform powered by AI.
								</p>
								<div className="flex flex-col sm:flex-row gap-6 justify-center">
									<Button
										variant="hero"
										size="hero"
										onClick={scrollToUpload}
										className="group"
									>
										Get Started Today
										<ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
									</Button>
									<Button
										variant="glass"
										size="hero"
										onClick={handleAboutClick}
									>
										Learn More
									</Button>
								</div>
							</div>
						</section>

						{/* Upload Section */}
						<UploadSection
							onFileUpload={handleFileUpload}
							onProcess={handleProcess}
							uploadedFile={uploadedFile}
							isProcessing={isProcessing}
						/>

						{/* Job Results Section */}
						{showJobs && (
							<section id="jobs-section" className="py-20 px-6 bg-muted/30">
								<div className="max-w-7xl mx-auto">
									<div className="text-center mb-16">
										<h2 className="text-4xl font-bold mb-4 text-primary">
											Your Matched Opportunities
										</h2>
										<p className="text-xl text-muted-foreground">
											Based on your profile, here are the top job matches for
											you
										</p>
									</div>

									<div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
										{jobs.map((job) => (
											<JobCard key={job.id} job={job} />
										))}
									</div>
								</div>
							</section>
						)}

						{/* Features Section */}
						{!showJobs && (
							<section className="py-20 px-6 bg-muted/30">
								<div className="max-w-6xl mx-auto">
									<div className="text-center mb-16">
										<h2 className="text-4xl font-bold mb-4">
											Why Choose CareerLink?
										</h2>
										<p className="text-xl text-muted-foreground">
											Discover the features that make us the leading career
											platform
										</p>
									</div>

									<div className="grid md:grid-cols-3 gap-8">
										{[
											{
												title: "Job Opportunities",
												description:
													"Discover thousands of job openings from top companies matched to your skills and career aspirations.",
											},
											{
												title: "Career Growth",
												description:
													"Access comprehensive tools and resources to accelerate your professional development journey.",
											},
											{
												title: "Networking",
												description:
													"Connect with industry professionals, mentors, and peers to expand your career opportunities.",
											},
										].map((feature, index) => (
											<Card
												key={index}
												className="p-10   border-2 border-primary/10 hover:border-primary/20  transition-all duration-500 hover:scale-105 group text-center"
											>
												<div className="mb-8">
													<h3 className="text-2xl font-sans font-bold mb-4 text-foreground group-hover:text-primary transition-all duration-300">
														{feature.title}
													</h3>
													<p className="text-muted-foreground leading-relaxed text-lg">
														{feature.description}
													</p>
												</div>
											</Card>
										))}
									</div>
								</div>
							</section>
						)}
					</>
				) : (
					// About Section
					<section className="py-32 px-6">
						<div className="max-w-4xl mx-auto">
							<div className="text-center mb-16">
								<h1 className="text-5xl font-bold mb-8 text-primary">
									About CareerLink
								</h1>
							</div>

							<Card className="p-12   border border-white/20 ">
								<div className="space-y-8">
									<p className="text-lg text-muted-foreground leading-relaxed">
										CareerLink is a revolutionary career platform designed to
										bridge the gap between talented professionals and exciting
										job opportunities. We leverage cutting-edge AI technology to
										provide personalized career solutions for job seekers at
										every stage of their journey.
									</p>

									<p className="text-lg text-muted-foreground leading-relaxed">
										From intelligent resume analysis to professional networking
										opportunities, CareerLink empowers users to manage their
										career journey with confidence and achieve their
										professional goals.
									</p>

									<div className="grid md:grid-cols-2 gap-8 mt-12">
										<div className="p-8   rounded-2xl border border-white/20">
											<div className="flex items-center gap-3 mb-4">
												<div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
													<Star className="w-6 h-6 text-white" />
												</div>
												<h3 className="text-2xl font-bold">Our Mission</h3>
											</div>
											<p className="text-muted-foreground leading-relaxed">
												To democratize access to career opportunities and
												connect talent with opportunity globally, regardless of
												background or location.
											</p>
										</div>

										<div className="p-8   rounded-2xl border border-white/20">
											<div className="flex items-center gap-3 mb-4">
												<div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
													<TrendingUp className="w-6 h-6 text-white" />
												</div>
												<h3 className="text-2xl font-bold">Our Vision</h3>
											</div>
											<p className="text-muted-foreground leading-relaxed">
												Creating a future where everyone has access to
												meaningful career opportunities and the tools to achieve
												their professional aspirations.
											</p>
										</div>
									</div>
								</div>
							</Card>
						</div>
					</section>
				)}
			</main>

			{/* Footer */}
			<footer className="border-t border-white/20 py-12 px-6">
				<div className="max-w-6xl mx-auto text-center">
					<div className="flex items-center justify-center space-x-2 mb-4">
						<div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
							<span className="text-white font-bold">C</span>
						</div>
						<span className="text-xl font-bold text-primary">CareerLink</span>
					</div>
					<p className="text-muted-foreground">
						© 2024 CareerLink. All rights reserved. Empowering careers
						worldwide.
					</p>
				</div>
			</footer>
		</div>
	);
};

export default Index;
