package main

// import (
// 	"context"
// 	"fmt"
// 	"log"
// 	"os"

// 	"github.com/fodedoumbouya/agent-sdk-go/pkg/model/providers/gemini"
// )

// func main() {
// 	// Get API key from environment
// 	apiKey := os.Getenv("GEMINI_API_KEY")
// 	if apiKey == "" {
// 		log.Fatal("GEMINI_API_KEY environment variable is required")
// 	}

// 	// Create a new Gemini provider
// 	provider := gemini.NewProvider(apiKey).
// 		WithDefaultModel("gemini-1.5-flash").
// 		WithRateLimit(60, 30000) // 60 requests per minute, 30k tokens per minute

// 	// Get a model
// 	model, err := provider.GetModel("gemini-1.5-flash")
// 	if err != nil {
// 		log.Fatalf("Failed to get model: %v", err)
// 	}

// 	// Create a simple request
// 	request := &model.Request{
// 		SystemInstructions: "You are a helpful assistant that provides concise and accurate answers.",
// 		Input:              "What is the capital of France?",
// 		Settings: &model.Settings{
// 			Temperature: func() *float64 { t := 0.7; return &t }(),
// 			MaxTokens:   func() *int { m := 100; return &m }(),
// 		},
// 	}

// 	// Get response
// 	fmt.Println("Getting response from Gemini...")
// 	response, err := model.GetResponse(context.Background(), request)
// 	if err != nil {
// 		log.Fatalf("Failed to get response: %v", err)
// 	}

// 	// Print the response
// 	fmt.Printf("Response: %s\n", response.Content)
// 	if response.Usage != nil {
// 		fmt.Printf("Token usage - Prompt: %d, Completion: %d, Total: %d\n",
// 			response.Usage.PromptTokens,
// 			response.Usage.CompletionTokens,
// 			response.Usage.TotalTokens)
// 	}

// 	// Example with function calling
// 	fmt.Println("\n--- Function Calling Example ---")

// 	// Define a simple function
// 	weatherTool := map[string]interface{}{
// 		"type": "function",
// 		"function": map[string]interface{}{
// 			"name":        "get_weather",
// 			"description": "Get the current weather for a location",
// 			"parameters": map[string]interface{}{
// 				"type": "object",
// 				"properties": map[string]interface{}{
// 					"location": map[string]interface{}{
// 						"type":        "string",
// 						"description": "The city and state, e.g. San Francisco, CA",
// 					},
// 				},
// 				"required": []string{"location"},
// 			},
// 		},
// 	}

// 	// Create request with tool
// 	toolRequest := &model.Request{
// 		SystemInstructions: "You are a helpful assistant that can get weather information.",
// 		Input:              "What's the weather like in New York?",
// 		Tools:              []interface{}{weatherTool},
// 		Settings: &model.Settings{
// 			Temperature: func() *float64 { t := 0.3; return &t }(),
// 		},
// 	}

// 	// Get response with potential tool calls
// 	toolResponse, err := model.GetResponse(context.Background(), toolRequest)
// 	if err != nil {
// 		log.Fatalf("Failed to get tool response: %v", err)
// 	}

// 	fmt.Printf("Response: %s\n", toolResponse.Content)
// 	if len(toolResponse.ToolCalls) > 0 {
// 		for _, toolCall := range toolResponse.ToolCalls {
// 			fmt.Printf("Tool call: %s with parameters: %v\n", toolCall.Name, toolCall.Parameters)
// 		}
// 	}

// 	// Example with streaming
// 	fmt.Println("\n--- Streaming Example ---")

// 	streamRequest := &model.Request{
// 		SystemInstructions: "You are a creative writing assistant.",
// 		Input:              "Write a short poem about coding in Go.",
// 		Settings: &model.Settings{
// 			Temperature: func() *float64 { t := 0.8; return &t }(),
// 		},
// 	}

// 	// Stream response
// 	streamChan, err := model.StreamResponse(context.Background(), streamRequest)
// 	if err != nil {
// 		log.Fatalf("Failed to start streaming: %v", err)
// 	}

// 	fmt.Println("Streaming response:")
// 	for event := range streamChan {
// 		switch event.Type {
// 		case model.StreamEventTypeContent:
// 			fmt.Print(event.Content)
// 		case model.StreamEventTypeToolCall:
// 			fmt.Printf("[Tool Call: %s]\n", event.ToolCall.Name)
// 		case model.StreamEventTypeDone:
// 			fmt.Println("\n[Stream completed]")
// 			if event.Response != nil && event.Response.Usage != nil {
// 				fmt.Printf("Token usage - Prompt: %d, Completion: %d, Total: %d\n",
// 					event.Response.Usage.PromptTokens,
// 					event.Response.Usage.CompletionTokens,
// 					event.Response.Usage.TotalTokens)
// 			}
// 		case model.StreamEventTypeError:
// 			fmt.Printf("[Error: %v]\n", event.Error)
// 		}
// 	}
// }
