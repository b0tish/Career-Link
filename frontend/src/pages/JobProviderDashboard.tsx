import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Navigation } from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import useAuthStore from "@/store/authStore";

const JobProviderDashboard = () => {
	const [jobs, setJobs] = useState([]);
	const [jobDetails, setJobDetails] = useState({
		job_title: "",
		company: "",
		location: "",
		salary: "",
		description: "",
		url: "",
	});
	const { token } = useAuthStore();

	useEffect(() => {
		fetchJobs();
	}, []);

	const fetchJobs = async () => {
		try {
			const response = await fetch("http://localhost:8000/api/jobs/", {
				headers: {
					Authorization: `Token ${token}`,
				},
			});
			if (response.ok) {
				const data = await response.json();
				setJobs(data);
			}
		} catch (error) {
			console.error("Error fetching jobs:", error);
		}
	};

	const handleInputChange = (
		e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
	) => {
		const { name, value } = e.target;
		setJobDetails({ ...jobDetails, [name]: value });
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		try {
			const response = await fetch("http://localhost:8000/api/jobs/", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Token ${token}`,
				},
				body: JSON.stringify(jobDetails),
			});
			if (response.ok) {
				fetchJobs();
				setJobDetails({
					job_title: "",
					company: "",
					location: "",
					salary: "",
					description: "",
					url: "",
				});
			}
		} catch (error) {
			console.error("Error posting job:", error);
		}
	};

	const handleDelete = async (jobId: string) => {
		try {
			const response = await fetch(`http://localhost:8000/api/jobs/${jobId}/`, {
				method: "DELETE",
				headers: {
					Authorization: `Token ${token}`,
				},
			});
			if (response.ok) {
				fetchJobs();
			}
		} catch (error) {
			console.error("Error deleting job:", error);
		}
	};

	return (
		<div className="min-h-screen bg-background flex flex-col">
			<Navigation />
			<div className="mt-8 container mx-auto p-4 pt-20">
				<h1 className="text-3xl font-bold mb-4">Job Provider Dashboard</h1>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
					<div>
						<Card>
							<CardHeader>
								<CardTitle>Post a New Job</CardTitle>
							</CardHeader>
							<CardContent>
								<form onSubmit={handleSubmit} className="space-y-4">
									<div>
										<Label htmlFor="job_title">Job Title</Label>
										<Input
											id="job_title"
											name="job_title"
											value={jobDetails.job_title}
											onChange={handleInputChange}
											required
										/>
									</div>
									<div>
										<Label htmlFor="company">Company</Label>
										<Input
											id="company"
											name="company"
											value={jobDetails.company}
											onChange={handleInputChange}
											required
										/>
									</div>
									<div>
										<Label htmlFor="location">Location</Label>
										<Input
											id="location"
											name="location"
											value={jobDetails.location}
											onChange={handleInputChange}
											required
										/>
									</div>
									<div>
										<Label htmlFor="salary">Salary</Label>
										<Input
											id="salary"
											name="salary"
											value={jobDetails.salary}
											onChange={handleInputChange}
										/>
									</div>
									<div>
										<Label htmlFor="description">Job Description</Label>
										<Textarea
											id="description"
											name="description"
											value={jobDetails.description}
											onChange={handleInputChange}
											required
										/>
									</div>
									<div>
										<Label htmlFor="url">Job URL</Label>
										<Input
											id="url"
											name="url"
											value={jobDetails.url}
											onChange={handleInputChange}
											required
										/>
									</div>
									<Button type="submit">Post Job</Button>
								</form>
							</CardContent>
						</Card>
					</div>
					<div>
						<Card>
							<CardHeader>
								<CardTitle>Your Posted Jobs</CardTitle>
							</CardHeader>
							<CardContent>
								{jobs.length > 0 ? (
									<ul className="space-y-4">
										{jobs.map((job: any) => (
											<li
												key={job._id}
												className="p-4 border rounded-md flex justify-between items-center"
											>
												<div>
													<h3 className="font-bold">{job.job_title}</h3>
													<p className="text-sm text-muted-foreground">
														{job.company}
													</p>
												</div>
												<Button
													variant="destructive"
													onClick={() => handleDelete(job._id)}
												>
													Delete
												</Button>
											</li>
										))}
									</ul>
								) : (
									<p>You haven't posted any jobs yet.</p>
								)}
							</CardContent>
						</Card>
					</div>
				</div>
			</div>
		</div>
	);
};

export default JobProviderDashboard;
