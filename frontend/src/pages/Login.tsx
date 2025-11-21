import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card'; // Import Card for styling
import useAuthStore from '@/store/authStore';
import { useNavigate, Link } from 'react-router-dom'; // Import Link
import { ArrowRight } from "lucide-react"; // Import for button icon
import { Navigation } from '@/components/Navigation'; // Import Navigation

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { setToken, setUser } = useAuthStore();
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (response.ok) {
        const data = await response.json();
        setToken(data.token);
        setUser(data.user);
        navigate('/');
      } else {
        // Use a more subtle feedback mechanism if possible, but keeping alert for now
        alert('Login failed: Invalid username or password');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('An error occurred during login');
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

      {/* Main Content Area - Centered Login */}
      <main className="flex flex-1 justify-center items-center py-24 px-6 pt-20">
        <Card className="w-full max-w-sm p-8 shadow-2xl border-2 border-primary/10 hover:border-primary/20 transition-all duration-300">
          <h2 className="text-3xl font-bold mb-8 text-center text-primary">Sign In to CareerLink</h2>

          <div className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="username">
                Username
              </Label>
              <Input
                id="username"
                type="text"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="bg-muted/30 border-primary/20 focus-visible:ring-primary"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">
                Password
              </Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-muted/30 border-primary/20 focus-visible:ring-primary"
              />
            </div>

            <div className="pt-4">
              <Button
                type="button"
                onClick={handleLogin}
                className="w-full group"
                size="lg" // Use a larger size for the main action
              >
                Log In
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </div>

            <p className="text-center text-sm text-muted-foreground">
              Don't have an account?
              <Link to="/register" className="text-primary hover:underline ml-1 font-medium">
                Sign Up
              </Link>
            </p>
          </div>
        </Card>
      </main>

      <Footer />
    </div>
  );
};

export default Login;
