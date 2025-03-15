package runner

import (
	"context"

	"github.com/pontus-devoteam/agent-sdk-go/pkg/agent"
	"github.com/pontus-devoteam/agent-sdk-go/pkg/result"
)

// SingleTurnResult contains the result of a single turn
type SingleTurnResult struct {
	Agent    *agent.Agent
	Response interface{}
	Output   interface{}
}

// RunHooks defines lifecycle hooks for a run
type RunHooks interface {
	// OnRunStart is called when the run starts
	OnRunStart(ctx context.Context, agent *agent.Agent, input interface{}) error

	// OnTurnStart is called when a turn starts
	OnTurnStart(ctx context.Context, agent *agent.Agent, turn int) error

	// OnTurnEnd is called when a turn ends
	OnTurnEnd(ctx context.Context, agent *agent.Agent, turn int, result *SingleTurnResult) error

	// OnRunEnd is called when the run ends
	OnRunEnd(ctx context.Context, result *result.RunResult) error
}

// DefaultRunHooks provides a default implementation of RunHooks
type DefaultRunHooks struct{}

// OnRunStart is called when the run starts
func (h *DefaultRunHooks) OnRunStart(ctx context.Context, agent *agent.Agent, input interface{}) error {
	return nil
}

// OnTurnStart is called when a turn starts
func (h *DefaultRunHooks) OnTurnStart(ctx context.Context, agent *agent.Agent, turn int) error {
	return nil
}

// OnTurnEnd is called when a turn ends
func (h *DefaultRunHooks) OnTurnEnd(ctx context.Context, agent *agent.Agent, turn int, result *SingleTurnResult) error {
	return nil
}

// OnRunEnd is called when the run ends
func (h *DefaultRunHooks) OnRunEnd(ctx context.Context, result *result.RunResult) error {
	return nil
} 