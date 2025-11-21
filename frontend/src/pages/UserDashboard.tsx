import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import useAuthStore from "@/store/authStore";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Navigation } from "@/components/Navigation";
import { useNavigate } from "react-router-dom";

const UserDashboard = () => {
	const { token } = useAuthStore();
	const [savedJobs, setSavedJobs] = useState([]);
	const [cvHistory, setCvHistory] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const navigate = useNavigate();

	useEffect(() => {
		const fetchDashboardData = async () => {
			try {
				const response = await fetch("/api/user-dashboard/", {
					headers: {
						Authorization: `Token ${token}`,
					},
				});

				if (!response.ok) {
					throw new Error("Failed to fetch dashboard data");
				}

				const data = await response.json();
				setSavedJobs(data.saved_jobs);
				setCvHistory(data.cv_history);
			} catch (err) {
				setError(err.message);
			} finally {
				setLoading(false);
			}
		};

		if (token) {
			fetchDashboardData();
		}
	}, [token]);

	const handleRemoveSavedJob = async (jobId: number) => {
		try {
			const response = await fetch(`/api/saved-jobs/${jobId}/`, {
				method: "DELETE",
				headers: {
					Authorization: `Token ${token}`,
				},
			});

			if (!response.ok) {
				throw new Error("Failed to delete saved job");
			}

			setSavedJobs(savedJobs.filter((job: any) => job.id !== jobId));
		} catch (err) {
			setError(err.message);
		}
	};

	const handleRemoveCv = async (cvId: number) => {
		try {
			const response = await fetch(`/api/cv-history/${cvId}/`, {
				method: "DELETE",
				headers: {
					Authorization: `Token ${token}`,
				},
			});

			if (!response.ok) {
				throw new Error("Failed to delete CV entry");
			}

			setCvHistory(cvHistory.filter((cv: any) => cv.id !== cvId));
		} catch (err) {
			setError(err.message);
		}
	};

	if (loading) {
		return <div>Loading...</div>;
	}

	if (error) {
		return <div>Error: {error}</div>;
	}

	return (
		<div className="min-h-screen bg-background flex flex-col">
			<Navigation />
			<div className="container mx-auto my-5 p-4 pt-20">
				<h1 className="text-3xl font-bold mb-4">User Dashboard</h1>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<Card>
						<CardHeader>
							<CardTitle>Saved Jobs</CardTitle>
						</CardHeader>
						<CardContent>
							<ul>
								{savedJobs.map((job: any) => {
									console.log("Saved Job:", job); // Debugging line
									return (
										<li key={job.id} className="mb-2 p-2 border rounded-md">
											<div className="flex justify-between items-center">
												<div>
													<p className="font-semibold">{job.job_title}</p>
													<p className="text-sm text-gray-500">{job.company_name}</p>
												</div>
												<div className="flex items-center space-x-2">
													{job.job_url && (
														<Button variant="hero" size="sm" asChild>
															<a href={job.job_url} target="_blank" rel="noopener noreferrer">
																Apply Now
															</a>
														</Button>
													)}
													<Button
														variant="destructive"
														size="sm"
														onClick={() => handleRemoveSavedJob(job.id)}
													>
														Remove
													</Button>
												</div>
											</div>
										</li>
									);
								})}
							</ul>
						</CardContent>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>CV History</CardTitle>
						</CardHeader>
						<CardContent>
							<ul>
								{cvHistory.map((cv: any) => (
									<li key={cv.id} className="mb-4">
										<>
											<div className="flex justify-between">
												<p className="font-semibold">{cv.file_name}</p>
												<div className="flex items-center space-x-2">
													<p className="text-sm text-gray-500">
														{new Date(cv.uploaded_at).toLocaleDateString()}
													</p>
													<Button
														variant="destructive"
														size="sm"
														onClick={() => handleRemoveCv(cv.id)}
													>
														Remove
													</Button>
												</div>
											</div>
											<div className="mt-2">
												{cv.skills.split(",").map((skill: string) => (
													<Badge key={skill} variant="secondary" className="mr-2">
														{skill.trim()}
													</Badge>
												))}
											</div>
										</>
									</li>
								))}
							</ul>
						</CardContent>
					</Card>
				</div>
			</div>
		</div>
	);
};

export default UserDashboard;
