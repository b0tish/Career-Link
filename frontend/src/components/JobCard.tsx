import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MapPin, Clock, DollarSign, Users } from "lucide-react";

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
		logo?: string;
	};
}

export function JobCard({ job }: JobCardProps) {
	return (
		<Card className="p-6 bg-gradient-card backdrop-blur-sm border border-white/20 hover:shadow-glass transition-all duration-300 hover:scale-105 group">
			<div className="flex items-start justify-between mb-4">
				<div className="flex items-center space-x-4">
					<div>
						<h3 className="font-bold text-lg text-foreground group-hover:text-primary transition-colors">
							{job.job_title}
						</h3>
						<p className="text-muted-foreground font-medium">{job.company}</p>
					</div>
				</div>
				<div className="text-right">
					<div className="inline-flex items-center px-3 py-1 rounded-full bg-accent/10 text-accent font-semibold text-sm">
						{Math.ceil(job.match * 100)}% Match
					</div>
				</div>
			</div>

			<p className="text-muted-foreground mb-4 line-clamp-2">
				{job.description}
			</p>

			<div className="flex flex-wrap gap-4 mb-4 text-sm text-muted-foreground">
				<div className="flex items-center gap-1">
					<MapPin className="w-4 h-4" />
					{job.location}
				</div>
				<div className="flex items-center gap-1">
					<Clock className="w-4 h-4" />
					{job.type}
				</div>
				<div className="flex items-center gap-1">
					<DollarSign className="w-4 h-4" />
					{job.salary}
				</div>
			</div>

			<div className="flex gap-3">
				<Button variant="hero" size="sm" className="flex-1">
					Apply Now
				</Button>
				<Button variant="outline" size="sm" className="flex-1">
					Save Job
				</Button>
			</div>
		</Card>
	);
}
