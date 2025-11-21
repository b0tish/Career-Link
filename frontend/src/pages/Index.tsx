import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Navigation } from "@/components/Navigation";
import { UploadSection } from "@/components/UploadSection";
import { JobCard } from "@/components/JobCard";
import { ArrowRight } from "lucide-react";
import useAuthStore from "@/store/authStore";
import { useNavigate } from "react-router-dom";

const Index = () => {
	const [uploadedFile, setUploadedFile] = useState<File | null>(null);
	const [isProcessing, setIsProcessing] = useState(false);
	const [showJobs, setShowJobs] = useState(false);
	const [jobs, setJobs] = useState([]);
	const [extractedSkills, setExtractedSkills] = useState<string[]>([]);
	const { isAuthenticated } = useAuthStore();
	const navigate = useNavigate();

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
		if (!isAuthenticated) {
			navigate('/login');
			return;
		}

		if (!uploadedFile) return;

		setIsProcessing(true);
		const formData = new FormData();
		formData.append("resume", uploadedFile);

		try {
			const response = await fetch("http://localhost:8000/api/upload/", {
				method: "POST",
				headers: {
					'Authorization': `Token ${useAuthStore.getState().token}`, // Add this line
				},
				body: formData,
			});

			if (response.ok) {
				const data = await response.json();
				setJobs(data.top_jobs);
				setExtractedSkills(data.extracted_skills);
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

	return (
		<div className="min-h-screen bg-background">
			<Navigation />

			{/* Main Content */}
			<main className="pt-20">
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
								onClick={() => navigate('/about')}
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
								<h3 className="text-2xl font-bold mt-8 mb-4 text-primary">Your Skills</h3>
								<p className="text-lg text-muted-foreground mb-4">These are the skills we extracted from your resume.</p>
								{extractedSkills.length > 0 && (
									<div className="flex flex-wrap justify-center gap-2 mb-4">
										{extractedSkills.map((skill: string) => (
											<Badge key={skill} variant="secondary">
												{skill}
											</Badge>
										))}
									</div>
								)}
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
						© 2025 CareerLink. All rights reserved. Empowering careers
						worldwide.
					</p>
				</div>
			</footer>
		</div>
	);
};

export default Index;
