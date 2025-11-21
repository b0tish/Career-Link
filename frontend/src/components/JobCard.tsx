import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MapPin } from "lucide-react";
import useAuthStore from "@/store/authStore";
import { toast } from "@/components/ui/use-toast";

interface JobCardProps {
	job: {
		id: string;
		job_title: string;
		company: string;
		location: string;
		type: string;
		salary: string;
		description: string;
		match: number;
		skill_score: number;
		keyword_score: number;
		logo?: string;
		skills: string;
		url:string
	};
}

export function JobCard({ job }: JobCardProps) {
	const { isAuthenticated, token } = useAuthStore();

	const handleSaveJob = async () => {
		try {
			const response = await fetch('/api/save-job/', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Token ${token}`,
				},
				body: JSON.stringify({
					job_title: job.job_title,
					company_name: job.company,
					location: job.location,
					job_description: job.description,
					skills: job.skills,
					job_url: job.url,
				}),
			});

			if (!response.ok) {
				throw new Error('Failed to save job');
			}

			toast({
				title: "Job Saved",
			});
		} catch (error) {
			toast({
				title: "Error",
				description: error.message,
			});
		}
	};

	return (
		<Card className="p-6 border transition-all duration-300 hover:scale-105 group">
			<div className="flex items-start justify-between mb-4">
				<div className="flex-grow">
					<h3 className="font-bold text-lg text-foreground group-hover:text-primary transition-colors">
						{job.job_title}
					</h3>
					<p className="text-muted-foreground font-medium">{job.company}</p>
				</div>
				<div className="text-right flex-shrink-0">
					<div className="inline-flex items-center px-3 py-1 rounded-full bg-accent text-accent-foreground font-semibold text-sm">
						<span>{Math.ceil(job.match * 100)}% Match</span>
					</div>
					<div className="text-xs text-muted-foreground mt-1">
						<span>Skill: {Math.ceil((job.skill_score || 0) * 100)}%</span>
						<span className="mx-1">|</span>
						<span>Keyword: {Math.ceil((job.keyword_score || 0) * 100)}%</span>
					</div>
				</div>
			</div>

			<p className="text-muted-foreground mb-4 line-clamp-2">
				{job.description}
			</p>

			<div className="flex flex-wrap gap-12 mb-4 text-sm text-muted-foreground">
				<div className="flex items-center gap-1">
					<MapPin className="w-4 h-4" />
					{job.location}
				</div>
				<div className="flex items-center gap-1">{job.type}</div>
				<div className="flex items-center gap-1">{job.salary}</div>
			</div>

			<div className="flex gap-3">
				<Button variant="hero" size="sm" className="flex-1">
					<a href={job.url} className="w-full">
						Apply Now
					</a>
				</Button>
				{isAuthenticated && (
					<Button variant="outline" size="sm" onClick={handleSaveJob}>
						Save
					</Button>
				)}
			</div>
		</Card>
	);
}
