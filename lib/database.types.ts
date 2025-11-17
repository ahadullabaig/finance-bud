/**
 * TypeScript types for Supabase database schema
 * Generated from the database schema for type safety
 */

export interface Database {
  public: {
    Tables: {
      user_profiles: {
        Row: {
          id: string
          email: string
          full_name: string | null
          risk_tolerance: string
          created_at: string
          updated_at: string
          metadata: any
        }
        Insert: {
          id: string
          email: string
          full_name?: string | null
          risk_tolerance?: string
          metadata?: any
        }
        Update: {
          id?: string
          email?: string
          full_name?: string | null
          risk_tolerance?: string
          metadata?: any
        }
      }
      financial_states: {
        Row: {
          id: string
          user_id: string
          total_assets: number
          total_liabilities: number
          net_worth: number
          monthly_income: number
          monthly_expenses: number
          monthly_cash_flow: number
          risk_tolerance: string
          tax_filing_status: string
          estimated_tax_rate: number
          as_of_date: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          total_assets?: number
          total_liabilities?: number
          monthly_income?: number
          monthly_expenses?: number
          risk_tolerance?: string
          tax_filing_status?: string
          estimated_tax_rate?: number
          as_of_date?: string
        }
        Update: {
          id?: string
          user_id?: string
          total_assets?: number
          total_liabilities?: number
          monthly_income?: number
          monthly_expenses?: number
          risk_tolerance?: string
          tax_filing_status?: string
          estimated_tax_rate?: number
          as_of_date?: string
        }
      }
      plans: {
        Row: {
          id: string
          user_id: string
          plan_data: any
          status: string
          confidence_score: number
          correlation_id: string
          session_id: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          plan_data: any
          status?: string
          confidence_score?: number
          correlation_id: string
          session_id?: string | null
        }
        Update: {
          id?: string
          user_id?: string
          plan_data?: any
          status?: string
          confidence_score?: number
          correlation_id?: string
          session_id?: string | null
        }
      }
      plan_steps: {
        Row: {
          id: string
          plan_id: string
          step_id: string
          sequence_number: number
          action_type: string
          description: string
          amount: number
          target_date: string
          rationale: string | null
          confidence_score: number
          risk_level: string
          status: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          plan_id: string
          step_id?: string
          sequence_number: number
          action_type: string
          description: string
          amount: number
          target_date: string
          rationale?: string | null
          confidence_score?: number
          risk_level?: string
          status?: string
        }
        Update: {
          id?: string
          plan_id?: string
          step_id?: string
          sequence_number?: number
          action_type?: string
          description?: string
          amount?: number
          target_date?: string
          rationale?: string | null
          confidence_score?: number
          risk_level?: string
          status?: string
        }
      }
      market_data: {
        Row: {
          id: string
          data_id: string
          timestamp: string
          source: string
          market_volatility: number
          interest_rates: any
          sector_trends: any
          economic_sentiment: number
          collection_method: string | null
          refresh_frequency: number
          created_at: string
        }
        Insert: {
          id?: string
          data_id?: string
          timestamp?: string
          source: string
          market_volatility?: number
          interest_rates?: any
          sector_trends?: any
          economic_sentiment?: number
          collection_method?: string | null
          refresh_frequency?: number
        }
        Update: {
          id?: string
          data_id?: string
          timestamp?: string
          source?: string
          market_volatility?: number
          interest_rates?: any
          sector_trends?: any
          economic_sentiment?: number
          collection_method?: string | null
          refresh_frequency?: number
        }
      }
      agent_messages: {
        Row: {
          id: string
          message_id: string
          agent_id: string
          target_agent_id: string | null
          message_type: string
          payload: any
          correlation_id: string
          session_id: string | null
          priority: string
          trace_id: string | null
          retry_count: number
          expires_at: string | null
          created_at: string
        }
        Insert: {
          id?: string
          message_id?: string
          agent_id: string
          target_agent_id?: string | null
          message_type: string
          payload: any
          correlation_id: string
          session_id?: string | null
          priority?: string
          trace_id?: string | null
          retry_count?: number
          expires_at?: string | null
        }
        Update: {
          id?: string
          message_id?: string
          agent_id?: string
          target_agent_id?: string | null
          message_type?: string
          payload?: any
          correlation_id?: string
          session_id?: string | null
          priority?: string
          trace_id?: string | null
          retry_count?: number
          expires_at?: string | null
        }
      }
      execution_logs: {
        Row: {
          id: string
          log_id: string
          agent_id: string
          action_type: string
          operation_name: string
          execution_status: string
          input_data: any
          output_data: any
          execution_time: number
          session_id: string | null
          correlation_id: string
          trace_id: string | null
          created_at: string
        }
        Insert: {
          id?: string
          log_id?: string
          agent_id: string
          action_type: string
          operation_name: string
          execution_status: string
          input_data?: any
          output_data?: any
          execution_time?: number
          session_id?: string | null
          correlation_id: string
          trace_id?: string | null
        }
        Update: {
          id?: string
          log_id?: string
          agent_id?: string
          action_type?: string
          operation_name?: string
          execution_status?: string
          input_data?: any
          output_data?: any
          execution_time?: number
          session_id?: string | null
          correlation_id?: string
          trace_id?: string | null
        }
      }
      verification_reports: {
        Row: {
          id: string
          report_id: string
          plan_id: string
          verification_status: string
          constraints_checked: number
          constraints_passed: number
          constraint_violations: any
          overall_risk_score: number
          approval_rationale: string | null
          confidence_score: number
          verification_time: number
          verifier_agent_id: string
          correlation_id: string
          created_at: string
        }
        Insert: {
          id?: string
          report_id?: string
          plan_id: string
          verification_status: string
          constraints_checked?: number
          constraints_passed?: number
          constraint_violations?: any
          overall_risk_score?: number
          approval_rationale?: string | null
          confidence_score?: number
          verification_time?: number
          verifier_agent_id: string
          correlation_id: string
        }
        Update: {
          id?: string
          report_id?: string
          plan_id?: string
          verification_status?: string
          constraints_checked?: number
          constraints_passed?: number
          constraint_violations?: any
          overall_risk_score?: number
          approval_rationale?: string | null
          confidence_score?: number
          verification_time?: number
          verifier_agent_id?: string
          correlation_id?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
  }
}