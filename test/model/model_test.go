package model_test

import (
	"testing"

	"github.com/fodedoumbouya/agent-sdk-go/pkg/model"
)

// TestRequest tests creating a model request
func TestRequest(t *testing.T) {
	// Create a model request
	systemInstructions := "System instructions"
	input := "User input"
	tools := []interface{}{
		map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "test_function",
				"description": "Test function description",
				"parameters": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"param1": map[string]interface{}{
							"type":        "string",
							"description": "Parameter 1",
						},
					},
					"required": []string{"param1"},
				},
			},
		},
	}
	outputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"result": map[string]interface{}{
				"type":        "string",
				"description": "Result of the operation",
			},
		},
		"required": []string{"result"},
	}

	handoffs := []interface{}{
		map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "handoff_function",
				"description": "Handoff function description",
				"parameters": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"param1": map[string]interface{}{
							"type":        "string",
							"description": "Parameter 1",
						},
					},
					"required": []string{"param1"},
				},
			},
		},
	}

	settings := &model.Settings{
		Temperature: func() *float64 { val := 0.7; return &val }(),
		TopP:        func() *float64 { val := 0.9; return &val }(),
	}

	req := &model.Request{
		SystemInstructions: systemInstructions,
		Input:              input,
		Tools:              tools,
		OutputSchema:       outputSchema,
		Handoffs:           handoffs,
		Settings:           settings,
	}

	// Check if request was created correctly
	if req.SystemInstructions != systemInstructions {
		t.Errorf("Request.SystemInstructions = %v, want %v", req.SystemInstructions, systemInstructions)
	}

	if req.Input != input {
		t.Errorf("Request.Input = %v, want %v", req.Input, input)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("len(Request.Tools) = %d, want 1", len(req.Tools))
	}

	tool0, ok := req.Tools[0].(map[string]interface{})
	if !ok {
		t.Fatalf("Request.Tools[0] is not a map[string]interface{}")
	}

	toolType, ok := tool0["type"].(string)
	if !ok || toolType != "function" {
		t.Errorf("Request.Tools[0].type = %v, want function", toolType)
	}

	function, ok := tool0["function"].(map[string]interface{})
	if !ok {
		t.Fatalf("Request.Tools[0].function is not a map[string]interface{}")
	}

	functionName, ok := function["name"].(string)
	if !ok || functionName != "test_function" {
		t.Errorf("Request.Tools[0].function.name = %v, want test_function", functionName)
	}

	if req.OutputSchema == nil {
		t.Fatalf("Request.OutputSchema is nil")
	}

	if len(req.Handoffs) != 1 {
		t.Fatalf("len(Request.Handoffs) = %d, want 1", len(req.Handoffs))
	}

	handoff0, ok := req.Handoffs[0].(map[string]interface{})
	if !ok {
		t.Fatalf("Request.Handoffs[0] is not a map[string]interface{}")
	}

	handoffFunction, ok := handoff0["function"].(map[string]interface{})
	if !ok {
		t.Fatalf("Request.Handoffs[0].function is not a map[string]interface{}")
	}

	handoffFunctionName, ok := handoffFunction["name"].(string)
	if !ok || handoffFunctionName != "handoff_function" {
		t.Errorf("Request.Handoffs[0].function.name = %v, want handoff_function", handoffFunctionName)
	}

	if req.Settings == nil {
		t.Fatalf("Request.Settings is nil")
	}

	if *req.Settings.Temperature != 0.7 {
		t.Errorf("Request.Settings.Temperature = %f, want 0.7", *req.Settings.Temperature)
	}

	if *req.Settings.TopP != 0.9 {
		t.Errorf("Request.Settings.TopP = %f, want 0.9", *req.Settings.TopP)
	}
}

// TestResponse tests creating a model response
func TestResponse(t *testing.T) {
	// Create a model response
	content := "Response content"
	toolCalls := []model.ToolCall{
		{
			ID:   "call_123",
			Name: "test_function",
			Parameters: map[string]interface{}{
				"param1": "value1",
			},
		},
	}
	handoffCall := &model.HandoffCall{
		AgentName:      "handoff_agent",
		Parameters:     map[string]any{"input": "Handoff input"},
		ReturnToAgent:  "",
		TaskID:         "",
		IsTaskComplete: false,
	}
	usage := &model.Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}

	resp := &model.Response{
		Content:     content,
		ToolCalls:   toolCalls,
		HandoffCall: handoffCall,
		Usage:       usage,
	}

	// Check if response was created correctly
	if resp.Content != content {
		t.Errorf("Response.Content = %v, want %v", resp.Content, content)
	}

	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(Response.ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}

	if resp.ToolCalls[0].ID != "call_123" {
		t.Errorf("Response.ToolCalls[0].ID = %s, want call_123", resp.ToolCalls[0].ID)
	}

	if resp.ToolCalls[0].Name != "test_function" {
		t.Errorf("Response.ToolCalls[0].Name = %s, want test_function", resp.ToolCalls[0].Name)
	}

	param1, ok := resp.ToolCalls[0].Parameters["param1"].(string)
	if !ok || param1 != "value1" {
		t.Errorf("Response.ToolCalls[0].Parameters[\"param1\"] = %v, want value1", resp.ToolCalls[0].Parameters["param1"])
	}

	if resp.HandoffCall == nil {
		t.Fatalf("Response.HandoffCall is nil")
	}

	if resp.HandoffCall.AgentName != "handoff_agent" {
		t.Errorf("Response.HandoffCall.AgentName = %s, want handoff_agent", resp.HandoffCall.AgentName)
	}

	input, ok := resp.HandoffCall.Parameters["input"].(string)
	if !ok || input != "Handoff input" {
		t.Errorf("Response.HandoffCall.Parameters[\"input\"] = %v, want Handoff input", resp.HandoffCall.Parameters["input"])
	}

	if resp.Usage == nil {
		t.Fatalf("Response.Usage is nil")
	}

	if resp.Usage.PromptTokens != 100 {
		t.Errorf("Response.Usage.PromptTokens = %d, want 100", resp.Usage.PromptTokens)
	}

	if resp.Usage.CompletionTokens != 50 {
		t.Errorf("Response.Usage.CompletionTokens = %d, want 50", resp.Usage.CompletionTokens)
	}

	if resp.Usage.TotalTokens != 150 {
		t.Errorf("Response.Usage.TotalTokens = %d, want 150", resp.Usage.TotalTokens)
	}
}

// TestToolCall tests creating a tool call
func TestToolCall(t *testing.T) {
	// Create a tool call
	id := "call_123"
	name := "test_function"
	parameters := map[string]interface{}{
		"param1": "value1",
		"param2": 42,
	}

	call := model.ToolCall{
		ID:         id,
		Name:       name,
		Parameters: parameters,
	}

	// Check if call was created correctly
	if call.ID != id {
		t.Errorf("ToolCall.ID = %s, want %s", call.ID, id)
	}

	if call.Name != name {
		t.Errorf("ToolCall.Name = %s, want %s", call.Name, name)
	}

	if len(call.Parameters) != 2 {
		t.Fatalf("len(ToolCall.Parameters) = %d, want 2", len(call.Parameters))
	}

	param1, ok := call.Parameters["param1"].(string)
	if !ok || param1 != "value1" {
		t.Errorf("ToolCall.Parameters[\"param1\"] = %v, want value1", call.Parameters["param1"])
	}

	param2, ok := call.Parameters["param2"].(int)
	if !ok || param2 != 42 {
		t.Errorf("ToolCall.Parameters[\"param2\"] = %v, want 42", call.Parameters["param2"])
	}
}

// TestStreamEvent tests creating a stream event
func TestStreamEvent(t *testing.T) {
	// Create a stream event
	eventType := model.StreamEventTypeContent
	content := "Stream content"
	toolCall := &model.ToolCall{
		ID:   "call_123",
		Name: "test_function",
		Parameters: map[string]interface{}{
			"param1": "value1",
		},
	}
	handoffCall := &model.HandoffCall{
		AgentName:      "handoff_agent",
		Parameters:     map[string]any{"input": "Handoff input"},
		ReturnToAgent:  "",
		TaskID:         "",
		IsTaskComplete: false,
	}
	response := &model.Response{
		Content: "Response content",
	}
	done := true
	err := error(nil)

	event := model.StreamEvent{
		Type:        eventType,
		Content:     content,
		ToolCall:    toolCall,
		HandoffCall: handoffCall,
		Done:        done,
		Error:       err,
		Response:    response,
	}

	// Check if event was created correctly
	if event.Type != eventType {
		t.Errorf("StreamEvent.Type = %v, want %v", event.Type, eventType)
	}

	if event.Content != content {
		t.Errorf("StreamEvent.Content = %s, want %s", event.Content, content)
	}

	if event.ToolCall == nil {
		t.Fatalf("StreamEvent.ToolCall is nil")
	}

	if event.ToolCall.ID != toolCall.ID {
		t.Errorf("StreamEvent.ToolCall.ID = %s, want %s", event.ToolCall.ID, toolCall.ID)
	}

	if event.HandoffCall == nil {
		t.Fatalf("StreamEvent.HandoffCall is nil")
	}

	if event.HandoffCall.AgentName != handoffCall.AgentName {
		t.Errorf("StreamEvent.HandoffCall.AgentName = %s, want %s", event.HandoffCall.AgentName, handoffCall.AgentName)
	}

	if event.Response == nil {
		t.Fatalf("StreamEvent.Response is nil")
	}

	if event.Response.Content != response.Content {
		t.Errorf("StreamEvent.Response.Content = %s, want %s", event.Response.Content, response.Content)
	}

	if !event.Done {
		t.Errorf("StreamEvent.Done = %v, want true", event.Done)
	}

	if event.Error != nil {
		t.Errorf("StreamEvent.Error = %v, want nil", event.Error)
	}
}

// TestSettings tests creating model settings
func TestSettings(t *testing.T) {
	// Create model settings
	temperature := 0.7
	topP := 0.9
	maxTokens := 2048
	presencePenalty := 0.0
	frequencyPenalty := 0.0
	toolChoice := "auto"
	parallelToolCalls := true

	settings := &model.Settings{
		Temperature:       func() *float64 { val := 0.7; return &val }(),
		TopP:              func() *float64 { val := 0.9; return &val }(),
		MaxTokens:         func() *int { val := 2048; return &val }(),
		PresencePenalty:   &presencePenalty,
		FrequencyPenalty:  &frequencyPenalty,
		ToolChoice:        &toolChoice,
		ParallelToolCalls: &parallelToolCalls,
	}

	// Check if settings were created correctly
	if *settings.Temperature != temperature {
		t.Errorf("Settings.Temperature = %f, want %f", *settings.Temperature, temperature)
	}

	if *settings.TopP != topP {
		t.Errorf("Settings.TopP = %f, want %f", *settings.TopP, topP)
	}

	if *settings.MaxTokens != maxTokens {
		t.Errorf("Settings.MaxTokens = %d, want %d", *settings.MaxTokens, maxTokens)
	}

	if *settings.PresencePenalty != presencePenalty {
		t.Errorf("Settings.PresencePenalty = %f, want %f", *settings.PresencePenalty, presencePenalty)
	}

	if *settings.FrequencyPenalty != frequencyPenalty {
		t.Errorf("Settings.FrequencyPenalty = %f, want %f", *settings.FrequencyPenalty, frequencyPenalty)
	}

	if *settings.ToolChoice != toolChoice {
		t.Errorf("Settings.ToolChoice = %s, want %s", *settings.ToolChoice, toolChoice)
	}

	if !*settings.ParallelToolCalls {
		t.Errorf("Settings.ParallelToolCalls = %v, want true", *settings.ParallelToolCalls)
	}
}
