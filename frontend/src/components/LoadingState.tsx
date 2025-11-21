/**
 * Loading State Component
 * 
 * Provides consistent loading states for async operations
 * Task 15 - Person D
 * Requirements: 11.4, 12.4
 */

import { motion } from 'motion/react';
import { Loader2, Activity } from 'lucide-react';
import { Card } from './ui/card';

interface LoadingStateProps {
  message?: string;
  variant?: 'default' | 'minimal' | 'fullscreen';
  showProgress?: boolean;
  progress?: number;
}

export function LoadingState({ 
  message = 'Processing...', 
  variant = 'default',
  showProgress = false,
  progress = 0
}: LoadingStateProps) {
  
  if (variant === 'minimal') {
    return (
      <div className="flex items-center justify-center gap-2 p-4">
        <Loader2 className="w-5 h-5 animate-spin text-[var(--color-cyan)]" />
        <span 
          className="text-sm text-[var(--color-blueprint)]"
          style={{ fontFamily: 'var(--font-mono)' }}
        >
          {message}
        </span>
      </div>
    );
  }

  if (variant === 'fullscreen') {
    return (
      <div className="fixed inset-0 bg-white/90 backdrop-blur-sm flex items-center justify-center z-50">
        <Card className="border-2 border-[var(--color-ink)] bg-white p-8 max-w-md">
          <div className="flex flex-col items-center gap-4">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            >
              <Activity className="w-12 h-12 text-[var(--color-cyan)]" />
            </motion.div>
            <div className="text-center">
              <div 
                className="data-label mb-2"
                style={{ color: 'var(--color-blueprint)' }}
              >
                SYSTEM_STATUS
              </div>
              <h3 
                className="text-xl mb-2"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--color-ink)' }}
              >
                {message}
              </h3>
              {showProgress && (
                <div className="w-full bg-[var(--color-grid)] h-2 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-[var(--color-cyan)] to-[var(--color-blueprint)]"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <Card className="border-2 border-[var(--color-ink)] bg-white p-6">
      <div className="flex items-center gap-4">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        >
          <Activity className="w-8 h-8 text-[var(--color-cyan)]" />
        </motion.div>
        <div className="flex-1">
          <div 
            className="data-label mb-1"
            style={{ color: 'var(--color-blueprint)' }}
          >
            PROCESSING
          </div>
          <p 
            className="text-lg"
            style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink)' }}
          >
            {message}
          </p>
          {showProgress && (
            <div className="mt-3">
              <div className="flex justify-between text-xs mb-1" style={{ fontFamily: 'var(--font-code)' }}>
                <span className="text-[var(--color-blueprint)]">Progress</span>
                <span className="text-[var(--color-cyan)]">{progress}%</span>
              </div>
              <div className="w-full bg-[var(--color-grid)] h-2 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-[var(--color-cyan)] to-[var(--color-blueprint)]"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <motion.div
      className={`bg-[var(--color-grid)] rounded ${className}`}
      animate={{
        opacity: [0.5, 1, 0.5],
      }}
      transition={{
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut"
      }}
    />
  );
}

export function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-3/4" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-5/6" />
      <div className="grid grid-cols-3 gap-4 mt-6">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    </div>
  );
}
