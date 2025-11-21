/**
 * Error Boundary Component
 * 
 * Provides graceful error handling for React components
 * Task 15 - Person D
 * Requirements: 11.4, 12.4
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Card className="border-2 border-[var(--color-vermillion)] bg-white p-8 m-4">
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-8 h-8 text-[var(--color-vermillion)] flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h2 
                className="text-2xl mb-2 text-[var(--color-vermillion)]" 
                style={{ fontFamily: 'var(--font-display)' }}
              >
                System Error Detected
              </h2>
              <p 
                className="text-[var(--color-blueprint)] mb-4"
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                An unexpected error occurred in the application. The system has been isolated to prevent cascading failures.
              </p>
              
              {this.state.error && (
                <div className="bg-[var(--color-paper)] border-2 border-[var(--color-grid)] p-4 mb-4">
                  <div 
                    className="data-label mb-2"
                    style={{ color: 'var(--color-vermillion)' }}
                  >
                    ERROR_MESSAGE
                  </div>
                  <code 
                    className="text-sm block"
                    style={{ fontFamily: 'var(--font-code)', color: 'var(--color-ink)' }}
                  >
                    {this.state.error.toString()}
                  </code>
                </div>
              )}

              <div className="flex gap-3">
                <Button
                  onClick={this.handleReset}
                  className="bg-[var(--color-ink)] text-[var(--color-cyan)] hover:bg-[var(--color-blueprint)] border-2 border-[var(--color-ink)]"
                  style={{ fontFamily: 'var(--font-mono)' }}
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  RESET_COMPONENT
                </Button>
                <Button
                  onClick={() => window.location.reload()}
                  variant="outline"
                  className="border-2 border-[var(--color-ink)]"
                  style={{ fontFamily: 'var(--font-mono)' }}
                >
                  RELOAD_PAGE
                </Button>
              </div>

              {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                <details className="mt-4">
                  <summary 
                    className="cursor-pointer text-sm mb-2"
                    style={{ fontFamily: 'var(--font-code)', color: 'var(--color-blueprint)' }}
                  >
                    Stack Trace (Development Only)
                  </summary>
                  <pre 
                    className="text-xs bg-[var(--color-paper)] border-2 border-[var(--color-grid)] p-4 overflow-auto max-h-64"
                    style={{ fontFamily: 'var(--font-code)' }}
                  >
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}
            </div>
          </div>
        </Card>
      );
    }

    return this.props.children;
  }
}
