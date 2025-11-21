import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Card } from "@/components/ui/card"; // Import Card for styling
import useAuthStore from "@/store/authStore";
import { useNavigate, Link } from "react-router-dom"; // Import Link
import { ArrowRight } from "lucide-react"; // Import for button icon
import { Navigation } from '@/components/Navigation'; // Import Navigation

const Register = () => {
	const [username, setUsername] = useState("");
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [role, setRole] = useState("user");
	const { setToken, setUser } = useAuthStore();
	const navigate = useNavigate();

	const handleRegister = async () => {
		try {
			const response = await fetch("/api/auth/register", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ username, email, password, role }),
			});

			if (response.ok) {
				const data = await response.json();
				setToken(data.token);
				setUser(data.user);
				navigate("/dashboard");
			} else {
				alert("Registration failed");
			}
		} catch (error) {
			console.error("Registration error:", error);
			alert("An error occurred during registration");
		}
	};

	// Simplified Footer component based on Index.tsx
	const Footer = () => (
		<footer className="border-t border-white/20 py-8 px-6">
			<div className="max-w-6xl mx-auto text-center">
				<p className="text-muted-foreground">
					© 2024 CareerLink. All rights reserved. Empowering careers worldwide.
				</p>
			</div>
		</footer>
	);

	return (
		<div className="min-h-screen bg-background flex flex-col">
			<Navigation />

			{/* Main Content Area - Centered Registration */}
			<main className="flex flex-1 justify-center items-center py-24 px-6 pt-20">
				<Card className="w-full max-w-md p-8 shadow-2xl border-2 border-primary/10 hover:border-primary/20 transition-all duration-300">
					<h2 className="text-3xl font-bold mb-8 text-center text-primary">
						Create Your CareerLink Account
					</h2>

					<div className="space-y-6">
						<div className="space-y-2">
							<Label htmlFor="username">Username</Label>
							<Input
								id="username"
								type="text"
								placeholder="Choose a username"
								value={username}
								onChange={(e) => setUsername(e.target.value)}
								className="bg-muted/30 border-primary/20 focus-visible:ring-primary"
							/>
						</div>

						<div className="space-y-2">
							<Label htmlFor="email">Email</Label>
							<Input
								id="email"
								type="email"
								placeholder="Enter your email address"
								value={email}
								onChange={(e) => setEmail(e.target.value)}
								className="bg-muted/30 border-primary/20 focus-visible:ring-primary"
							/>
						</div>

						<div className="space-y-2">
							<Label htmlFor="password">Password</Label>
							<Input
								id="password"
								type="password"
								placeholder="Create a password"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								className="bg-muted/30 border-primary/20 focus-visible:ring-primary"
							/>
						</div>

						<div className="space-y-2">
							<Label className="text-foreground font-semibold">
								I am registering as a...
							</Label>
							<RadioGroup
								defaultValue="user"
								onValueChange={setRole}
								className="flex space-x-4 pt-2"
							>
								<div className="flex items-center space-x-2">
									<RadioGroupItem
										value="user"
										id="user"
										className="text-primary"
									/>
									<Label htmlFor="user">Job Seeker</Label>
								</div>
								<div className="flex items-center space-x-2">
									<RadioGroupItem
										value="job_provider"
										id="job_provider"
										className="text-primary"
									/>
									<Label htmlFor="job_provider">Job Provider</Label>
								</div>
							</RadioGroup>
						</div>

						<div className="pt-4">
							<Button
								type="button"
								onClick={handleRegister}
								className="w-full group"
								size="lg" // Use a larger size for the main action
							>
								Sign Up
								<ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
							</Button>
						</div>

						<p className="text-center text-sm text-muted-foreground">
							Already have an account?
							<Link
								to="/login"
								className="text-primary hover:underline ml-1 font-medium"
							>
								Log In
							</Link>
						</p>
					</div>
				</Card>
			</main>

			<Footer />
		</div>
	);
};

export default Register;
