/**
 * Error Message Component
 * 
 * User-friendly error messages with recovery options
 * Task 15 - Person D
 * Requirements: 11.4, 12.4
 */

import { AlertTriangle, XCircle, AlertCircle, RefreshCw, Home } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface ErrorMessageProps {
  title?: string;
  message: string;
  severity?: 'error' | 'warning' | 'info';
  onRetry?: () => void;
  onDismiss?: () => void;
  showHomeButton?: boolean;
  technicalDetails?: string;
}

export function ErrorMessage({
  title,
  message,
  severity = 'error',
  onRetry,
  onDismiss,
  showHomeButton = false,
  technicalDetails
}: ErrorMessageProps) {
  
  const getIcon = () => {
    switch (severity) {
      case 'error':
        return <XCircle className="w-6 h-6" />;
      case 'warning':
        return <AlertTriangle className="w-6 h-6" />;
      case 'info':
        return <AlertCircle className="w-6 h-6" />;
    }
  };

  const getColors = () => {
    switch (severity) {
      case 'error':
        return {
          border: 'border-[var(--color-vermillion)]',
          text: 'text-[var(--color-vermillion)]',
          bg: 'bg-[var(--color-vermillion)]/5'
        };
      case 'warning':
        return {
          border: 'border-[var(--color-amber)]',
          text: 'text-[var(--color-amber)]',
          bg: 'bg-[var(--color-amber)]/5'
        };
      case 'info':
        return {
          border: 'border-[var(--color-cyan)]',
          text: 'text-[var(--color-cyan)]',
          bg: 'bg-[var(--color-cyan)]/5'
        };
    }
  };

  const colors = getColors();
  const defaultTitle = severity === 'error' ? 'Error' : 
                      severity === 'warning' ? 'Warning' : 'Information';

  return (
    <Card className={`border-2 ${colors.border} ${colors.bg} p-6`}>
      <div className="flex items-start gap-4">
        <div className={`${colors.text} flex-shrink-0 mt-1`}>
          {getIcon()}
        </div>
        
        <div className="flex-1">
          <h3 
            className={`text-lg mb-2 ${colors.text}`}
            style={{ fontFamily: 'var(--font-display)' }}
          >
            {title || defaultTitle}
          </h3>
          
          <p 
            className="text-[var(--color-ink)] mb-4"
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem' }}
          >
            {message}
          </p>

          {technicalDetails && (
            <details className="mb-4">
              <summary 
                className="cursor-pointer text-sm mb-2"
                style={{ fontFamily: 'var(--font-code)', color: 'var(--color-blueprint)' }}
              >
                Technical Details
              </summary>
              <div className="bg-[var(--color-paper)] border-2 border-[var(--color-grid)] p-3 rounded">
                <code 
                  className="text-xs block whitespace-pre-wrap"
                  style={{ fontFamily: 'var(--font-code)', color: 'var(--color-ink)' }}
                >
                  {technicalDetails}
                </code>
              </div>
            </details>
          )}

          <div className="flex gap-3">
            {onRetry && (
              <Button
                onClick={onRetry}
                className="bg-[var(--color-ink)] text-[var(--color-cyan)] hover:bg-[var(--color-blueprint)] border-2 border-[var(--color-ink)]"
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                RETRY
              </Button>
            )}
            
            {showHomeButton && (
              <Button
                onClick={() => window.location.href = '/'}
                variant="outline"
                className="border-2 border-[var(--color-ink)]"
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                <Home className="w-4 h-4 mr-2" />
                HOME
              </Button>
            )}
            
            {onDismiss && (
              <Button
                onClick={onDismiss}
                variant="outline"
                className="border-2 border-[var(--color-ink)]"
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                DISMISS
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

// Common error messages
export const ErrorMessages = {
  NETWORK_ERROR: {
    title: 'Network Connection Error',
    message: 'Unable to connect to the server. Please check your internet connection and try again.',
    severity: 'error' as const
  },
  API_ERROR: {
    title: 'API Request Failed',
    message: 'The server encountered an error processing your request. Our team has been notified.',
    severity: 'error' as const
  },
  VALIDATION_ERROR: {
    title: 'Validation Error',
    message: 'Please check your input and try again. Some required fields may be missing or invalid.',
    severity: 'warning' as const
  },
  TIMEOUT_ERROR: {
    title: 'Request Timeout',
    message: 'The request took too long to complete. Please try again.',
    severity: 'warning' as const
  },
  UNAUTHORIZED: {
    title: 'Authentication Required',
    message: 'You need to be logged in to access this feature.',
    severity: 'warning' as const
  },
  NOT_FOUND: {
    title: 'Resource Not Found',
    message: 'The requested resource could not be found. It may have been moved or deleted.',
    severity: 'info' as const
  }
};
