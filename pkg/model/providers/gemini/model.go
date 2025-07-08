package gemini

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"math"
	mathrand "math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/fodedoumbouya/agent-sdk-go/pkg/model"
)

// Model implements the model.Model interface for Google Gemini
type Model struct {
	// Configuration
	ModelName string
	Provider  *Provider
}

// GeminiRequest represents a request to the Gemini API
type GeminiRequest struct {
	Contents          []Content         `json:"contents"`
	Tools             []Tool            `json:"tools,omitempty"`
	ToolConfig        *ToolConfig       `json:"toolConfig,omitempty"`
	SafetySettings    []SafetySetting   `json:"safetySettings,omitempty"`
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
}

// Content represents content in a Gemini request/response
type Content struct {
	Parts []Part `json:"parts"`
	Role  string `json:"role,omitempty"`
}

// Part represents a part of content
type Part struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
}

// FunctionCall represents a function call
type FunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

// FunctionResponse represents a function response
type FunctionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

// Tool represents a tool in a Gemini request
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations"`
}

// FunctionDeclaration represents a function declaration
type FunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolConfig represents tool configuration
type ToolConfig struct {
	FunctionCallingConfig *FunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// FunctionCallingConfig represents function calling configuration
type FunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// SafetySetting represents a safety setting
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GenerationConfig represents generation configuration
type GenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

// GeminiResponse represents a response from the Gemini API
type GeminiResponse struct {
	Candidates     []Candidate     `json:"candidates"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
}

// Candidate represents a candidate in a Gemini response
type Candidate struct {
	Content       Content        `json:"content"`
	FinishReason  string         `json:"finishReason,omitempty"`
	Index         int            `json:"index,omitempty"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// SafetyRating represents a safety rating
type SafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

// UsageMetadata represents usage metadata
type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// PromptFeedback represents prompt feedback
type PromptFeedback struct {
	BlockReason   string         `json:"blockReason,omitempty"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// ErrorResponse represents an error response from the Gemini API
type ErrorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// GetResponse gets a single response from the model with retry logic
func (m *Model) GetResponse(ctx context.Context, request *model.Request) (*model.Response, error) {
	var response *model.Response
	var lastErr error

	// Try with exponential backoff
	for attempt := 0; attempt <= m.Provider.MaxRetries; attempt++ {
		// Wait for rate limit
		m.Provider.WaitForRateLimit()

		// If this is not the first attempt, wait with exponential backoff
		if attempt > 0 {
			backoffDuration := calculateBackoff(attempt, m.Provider.RetryAfter)
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("context cancelled during backoff: %w", ctx.Err())
			case <-time.After(backoffDuration):
				// Continue after backoff
			}
		}

		// Try to get a response
		response, lastErr = m.getResponseOnce(ctx, request)

		// If successful or not a rate limit error, return
		if lastErr == nil {
			return response, nil
		}

		// If it's not a rate limit error, don't retry
		if !isRateLimitError(lastErr) {
			return nil, lastErr
		}

		// If we've exceeded the maximum number of retries, return the last error
		if attempt == m.Provider.MaxRetries {
			return nil, fmt.Errorf("exceeded maximum number of retries (%d): %w", m.Provider.MaxRetries, lastErr)
		}
	}

	// This should never happen
	return nil, lastErr
}

// getResponseOnce attempts to get a response from the model once
func (m *Model) getResponseOnce(ctx context.Context, request *model.Request) (*model.Response, error) {
	// Construct the request
	geminiRequest, err := m.constructRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to construct request: %w", err)
	}

	// Marshal the request to JSON
	requestBody, err := json.Marshal(geminiRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create the HTTP request
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", m.Provider.BaseURL, m.ModelName, m.Provider.APIKey)
	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		url,
		bytes.NewReader(requestBody),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	httpRequest.Header.Set("Content-Type", "application/json")

	// Send the request
	httpResponse, err := m.Provider.HTTPClient.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer func() {
		if closeErr := httpResponse.Body.Close(); closeErr != nil {
			// If we already have an error, keep it as the primary error
			if err == nil {
				err = fmt.Errorf("error closing response body: %w", closeErr)
			}
		}
	}()

	// Check for errors
	if httpResponse.StatusCode != http.StatusOK {
		return nil, m.handleError(httpResponse)
	}

	// Read the response
	responseBody, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Unmarshal the response
	var geminiResponse GeminiResponse
	if err := json.Unmarshal(responseBody, &geminiResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Update token count for rate limiting
	if geminiResponse.UsageMetadata != nil && geminiResponse.UsageMetadata.TotalTokenCount > 0 {
		m.Provider.UpdateTokenCount(geminiResponse.UsageMetadata.TotalTokenCount)
	}

	// Parse the response
	return m.parseResponse(&geminiResponse)
}

// StreamResponse streams a response from the model with retry logic
func (m *Model) StreamResponse(ctx context.Context, request *model.Request) (<-chan model.StreamEvent, error) {
	// Create a channel for stream events
	eventChan := make(chan model.StreamEvent)

	go func() {
		defer close(eventChan)

		var lastErr error

		// Try with exponential backoff
		for attempt := 0; attempt <= m.Provider.MaxRetries; attempt++ {
			// Wait for rate limit
			m.Provider.WaitForRateLimit()

			// If this is not the first attempt, wait with exponential backoff
			if attempt > 0 {
				backoffDuration := calculateBackoff(attempt, m.Provider.RetryAfter)
				select {
				case <-ctx.Done():
					eventChan <- model.StreamEvent{
						Type:  model.StreamEventTypeError,
						Error: fmt.Errorf("context cancelled during backoff: %w", ctx.Err()),
					}
					return
				case <-time.After(backoffDuration):
					// Continue after backoff
				}
			}

			// Try to stream a response
			err := m.streamResponseOnce(ctx, request, eventChan)

			// If successful, return
			if err == nil {
				return
			}

			lastErr = err

			// If it's not a rate limit error or context is cancelled, don't retry
			if !isRateLimitError(err) || ctx.Err() != nil {
				eventChan <- model.StreamEvent{
					Type:  model.StreamEventTypeError,
					Error: err,
				}
				return
			}

			// If we've exceeded the maximum number of retries, return the last error
			if attempt == m.Provider.MaxRetries {
				eventChan <- model.StreamEvent{
					Type:  model.StreamEventTypeError,
					Error: fmt.Errorf("exceeded maximum number of retries (%d): %w", m.Provider.MaxRetries, lastErr),
				}
				return
			}

			// Inform the caller that we're retrying
			eventChan <- model.StreamEvent{
				Type:    model.StreamEventTypeContent,
				Content: fmt.Sprintf("\n[Rate limit exceeded, retrying (attempt %d/%d)]", attempt+1, m.Provider.MaxRetries),
			}
		}
	}()

	return eventChan, nil
}

// streamResponseOnce attempts to stream a response from the model once
func (m *Model) streamResponseOnce(ctx context.Context, request *model.Request, eventChan chan<- model.StreamEvent) error {
	// Construct the request
	geminiRequest, err := m.constructRequest(request)
	if err != nil {
		return fmt.Errorf("failed to construct request: %w", err)
	}

	// Marshal the request to JSON
	requestBody, err := json.Marshal(geminiRequest)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create the HTTP request for streaming
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?key=%s", m.Provider.BaseURL, m.ModelName, m.Provider.APIKey)
	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		url,
		bytes.NewReader(requestBody),
	)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	httpRequest.Header.Set("Content-Type", "application/json")

	// Send the request
	httpResponse, err := m.Provider.HTTPClient.Do(httpRequest)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer func() {
		if closeErr := httpResponse.Body.Close(); closeErr != nil {
			eventChan <- model.StreamEvent{
				Type:  model.StreamEventTypeError,
				Error: fmt.Errorf("error closing response body: %w", closeErr),
			}
		}
	}()

	// Check for errors
	if httpResponse.StatusCode != http.StatusOK {
		return m.handleError(httpResponse)
	}

	// Read the streaming response
	scanner := bufio.NewScanner(httpResponse.Body)
	var responseBuilder strings.Builder
	var toolCalls []model.ToolCall
	var handoffCall *model.HandoffCall
	var usage *model.Usage

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines
		if strings.TrimSpace(line) == "" {
			continue
		}

		// Handle Server-Sent Events format if used
		line = strings.TrimPrefix(line, "data: ")

		// Parse the JSON line
		var geminiResponse GeminiResponse
		if err := json.Unmarshal([]byte(line), &geminiResponse); err != nil {
			// Skip lines that are not valid JSON
			continue
		}

		// Update token count for rate limiting
		if geminiResponse.UsageMetadata != nil && geminiResponse.UsageMetadata.TotalTokenCount > 0 {
			m.Provider.UpdateTokenCount(geminiResponse.UsageMetadata.TotalTokenCount)
			usage = &model.Usage{
				PromptTokens:     geminiResponse.UsageMetadata.PromptTokenCount,
				CompletionTokens: geminiResponse.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      geminiResponse.UsageMetadata.TotalTokenCount,
			}
		}

		// Process each candidate
		for _, candidate := range geminiResponse.Candidates {
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					responseBuilder.WriteString(part.Text)
					eventChan <- model.StreamEvent{
						Type:    model.StreamEventTypeContent,
						Content: part.Text,
					}
				}

				if part.FunctionCall != nil {
					toolCall := model.ToolCall{
						ID:         generateID(),
						Name:       part.FunctionCall.Name,
						Parameters: part.FunctionCall.Args,
					}
					toolCalls = append(toolCalls, toolCall)
					eventChan <- model.StreamEvent{
						Type:     model.StreamEventTypeToolCall,
						ToolCall: &toolCall,
					}
				}
			}

			// Check if this is the final response
			if candidate.FinishReason != "" {
				eventChan <- model.StreamEvent{
					Type: model.StreamEventTypeDone,
					Done: true,
					Response: &model.Response{
						Content:     responseBuilder.String(),
						ToolCalls:   toolCalls,
						HandoffCall: handoffCall,
						Usage:       usage,
					},
				}
				return nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading streaming response: %w", err)
	}

	// Send final done event if not already sent
	eventChan <- model.StreamEvent{
		Type: model.StreamEventTypeDone,
		Done: true,
		Response: &model.Response{
			Content:     responseBuilder.String(),
			ToolCalls:   toolCalls,
			HandoffCall: handoffCall,
			Usage:       usage,
		},
	}

	return nil
}

// constructRequest constructs a Gemini request from a model request
func (m *Model) constructRequest(request *model.Request) (*GeminiRequest, error) {
	geminiRequest := &GeminiRequest{}

	// Add system instruction if provided
	if request.SystemInstructions != "" {
		geminiRequest.SystemInstruction = &Content{
			Parts: []Part{{Text: request.SystemInstructions}},
		}
	}

	// Convert input to content
	var inputText string
	switch input := request.Input.(type) {
	case string:
		inputText = input
	case []interface{}:
		// Handle conversation history
		for _, msg := range input {
			if msgMap, ok := msg.(map[string]interface{}); ok {
				role, _ := msgMap["role"].(string)
				content, _ := msgMap["content"].(string)

				geminiContent := Content{
					Parts: []Part{{Text: content}},
				}

				// Map roles to Gemini format
				switch role {
				case "user":
					geminiContent.Role = "user"
				case "assistant":
					geminiContent.Role = "model"
				case "system":
					// System messages are handled via SystemInstruction
					continue
				default:
					geminiContent.Role = "user"
				}

				geminiRequest.Contents = append(geminiRequest.Contents, geminiContent)
			}
		}
	default:
		return nil, fmt.Errorf("unsupported input type: %T", request.Input)
	}

	// If we have simple string input, add it as user content
	if inputText != "" {
		geminiRequest.Contents = append(geminiRequest.Contents, Content{
			Parts: []Part{{Text: inputText}},
			Role:  "user",
		})
	}

	// Add tools if provided
	if len(request.Tools) > 0 {
		var functionDeclarations []FunctionDeclaration
		for _, tool := range request.Tools {
			if toolMap, ok := tool.(map[string]interface{}); ok {
				if funcMap, ok := toolMap["function"].(map[string]interface{}); ok {
					name, _ := funcMap["name"].(string)
					description, _ := funcMap["description"].(string)
					parameters, _ := funcMap["parameters"].(map[string]interface{})

					functionDeclarations = append(functionDeclarations, FunctionDeclaration{
						Name:        name,
						Description: description,
						Parameters:  parameters,
					})
				}
			}
		}

		if len(functionDeclarations) > 0 {
			geminiRequest.Tools = []Tool{{
				FunctionDeclarations: functionDeclarations,
			}}
		}
	}

	// Add generation config based on settings
	if request.Settings != nil {
		config := &GenerationConfig{}
		if request.Settings.Temperature != nil {
			config.Temperature = request.Settings.Temperature
		}
		if request.Settings.TopP != nil {
			config.TopP = request.Settings.TopP
		}
		if request.Settings.MaxTokens != nil {
			config.MaxOutputTokens = request.Settings.MaxTokens
		}
		geminiRequest.GenerationConfig = config
	}

	return geminiRequest, nil
}

// parseResponse parses a Gemini response into a model response
func (m *Model) parseResponse(geminiResponse *GeminiResponse) (*model.Response, error) {
	if len(geminiResponse.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in response")
	}

	candidate := geminiResponse.Candidates[0]
	response := &model.Response{}

	// Extract content and tool calls
	var contentBuilder strings.Builder
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			contentBuilder.WriteString(part.Text)
		}

		if part.FunctionCall != nil {
			response.ToolCalls = append(response.ToolCalls, model.ToolCall{
				ID:         generateID(),
				Name:       part.FunctionCall.Name,
				Parameters: part.FunctionCall.Args,
			})
		}
	}

	response.Content = contentBuilder.String()

	// Extract usage information
	if geminiResponse.UsageMetadata != nil {
		response.Usage = &model.Usage{
			PromptTokens:     geminiResponse.UsageMetadata.PromptTokenCount,
			CompletionTokens: geminiResponse.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      geminiResponse.UsageMetadata.TotalTokenCount,
		}
	}

	return response, nil
}

// handleError handles error responses from the API
func (m *Model) handleError(response *http.Response) error {
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf("HTTP %d: failed to read error response: %w", response.StatusCode, err)
	}

	var errorResponse ErrorResponse
	if err := json.Unmarshal(body, &errorResponse); err != nil {
		return fmt.Errorf("HTTP %d: %s", response.StatusCode, string(body))
	}

	return fmt.Errorf("HTTP %d: %s", response.StatusCode, errorResponse.Error.Message)
}

// isRateLimitError checks if an error is a rate limit error
func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	return strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "429") ||
		strings.Contains(errStr, "quota")
}

// calculateBackoff calculates the backoff duration for retries
func calculateBackoff(attempt int, baseDelay time.Duration) time.Duration {
	// Exponential backoff with jitter
	backoff := time.Duration(math.Pow(2, float64(attempt))) * baseDelay
	jitter := time.Duration(mathrand.Int63n(int64(backoff) / 10)) // 10% jitter
	return backoff + jitter
}

// generateID generates a random ID for tool calls
func generateID() string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, 16)
	rand.Read(b)
	for i := range b {
		b[i] = charset[b[i]%byte(len(charset))]
	}
	return string(b)
}
