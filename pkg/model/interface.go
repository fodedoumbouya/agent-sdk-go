package model

import (
	"context"
)

// Request represents a request to a model
type Request struct {
	SystemInstructions string
	Input              interface{}
	Tools              []interface{}
	OutputSchema       interface{}
	Handoffs           []interface{}
	Settings           *Settings
}

// Response represents a response from a model
type Response struct {
	Content     string
	ToolCalls   []ToolCall
	HandoffCall *HandoffCall
	Usage       *Usage
}

// ToolCall represents a tool call from a model
type ToolCall struct {
	ID         string
	Name       string
	Parameters map[string]interface{}
}

// HandoffCall represents a handoff call from a model
type HandoffCall struct {
	AgentName string
	Input     string
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// StreamEvent represents an event in a streaming response
type StreamEvent struct {
	Type        string
	Content     string
	ToolCall    *ToolCall
	HandoffCall *HandoffCall
	Done        bool
	Error       error
	Response    *Response
}

// StreamEvent types
const (
	StreamEventTypeContent  = "content"
	StreamEventTypeToolCall = "tool_call"
	StreamEventTypeHandoff  = "handoff"
	StreamEventTypeDone     = "done"
	StreamEventTypeError    = "error"
)

// Settings configures model-specific parameters
type Settings struct {
	Temperature       *float64
	TopP              *float64
	FrequencyPenalty  *float64
	PresencePenalty   *float64
	ToolChoice        *string
	ParallelToolCalls *bool
	MaxTokens         *int
}

// Model defines the interface for interacting with LLMs
type Model interface {
	// GetResponse gets a single response from the model
	GetResponse(ctx context.Context, request *Request) (*Response, error)

	// StreamResponse streams a response from the model
	StreamResponse(ctx context.Context, request *Request) (<-chan StreamEvent, error)
}

// Provider is responsible for looking up Models by name
type Provider interface {
	// GetModel returns a model by name
	GetModel(modelName string) (Model, error)
}
