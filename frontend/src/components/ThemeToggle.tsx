import { Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';
import { Button } from './ui/button';

export function ThemeToggle() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // useEffect only runs on the client, so now we can safely show the UI
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Button
        variant="outline"
        size="icon"
        className="border-2 border-[var(--color-ink)] bg-[var(--color-paper)] text-[var(--color-ink)] hover:bg-[var(--color-ink)] hover:text-[var(--color-cyan)] transition-all shrink-0 shadow-lg"
      >
        <Sun className="h-[1.2rem] w-[1.2rem]" />
        <span className="sr-only">Toggle theme</span>
      </Button>
    );
  }

  const isDark = theme === 'dark' || resolvedTheme === 'dark';

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className="border-2 border-[var(--color-ink)] bg-[var(--color-paper)] text-[var(--color-ink)] hover:bg-[var(--color-ink)] hover:text-[var(--color-cyan)] transition-all shrink-0 shadow-lg backdrop-blur-sm"
      title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
    >
      {isDark ? (
        <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-transform duration-300 hover:rotate-180" />
      ) : (
        <Moon className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-transform duration-300 hover:rotate-12" />
      )}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}
