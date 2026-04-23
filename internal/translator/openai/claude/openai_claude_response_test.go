package claude

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestExtractOpenAIUsage_DoesNotSubtractCachedTokensFromPrompt(t *testing.T) {
	usage := gjson.Parse(`{"prompt_tokens":17512,"completion_tokens":30,"prompt_tokens_details":{"cached_tokens":17500}}`)

	inputTokens, outputTokens, cachedTokens := extractOpenAIUsage(usage)

	if inputTokens != 17512 {
		t.Fatalf("expected prompt_tokens %d, got %d", 17512, inputTokens)
	}
	if outputTokens != 30 {
		t.Fatalf("expected completion_tokens %d, got %d", 30, outputTokens)
	}
	if cachedTokens != 17500 {
		t.Fatalf("expected cached_tokens %d, got %d", 17500, cachedTokens)
	}
}

func TestConvertOpenAIResponseToClaudeNonStream_PreservesPromptTokensWithCachedUsage(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	response := []byte(`{
		"id":"chatcmpl_cached",
		"object":"chat.completion",
		"created":1770000000,
		"model":"gpt-5.4-high",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"ok"},
			"finish_reason":"stop"
		}],
		"usage":{
			"prompt_tokens":17512,
			"completion_tokens":30,
			"total_tokens":17542,
			"prompt_tokens_details":{"cached_tokens":17500}
		}
	}`)

	out := ConvertOpenAIResponseToClaudeNonStream(ctx, "", originalRequest, nil, response, nil)
	parsed := gjson.ParseBytes(out)

	if got := parsed.Get("usage.input_tokens").Int(); got != 17512 {
		t.Fatalf("expected usage.input_tokens %d, got %d", 17512, got)
	}
	if got := parsed.Get("usage.output_tokens").Int(); got != 30 {
		t.Fatalf("expected usage.output_tokens %d, got %d", 30, got)
	}
	if got := parsed.Get("usage.cache_read_input_tokens").Int(); got != 17500 {
		t.Fatalf("expected usage.cache_read_input_tokens %d, got %d", 17500, got)
	}
}

func TestExtractOpenAIUsage_SupportsResponsesStyleUsage(t *testing.T) {
	usage := gjson.Parse(`{"input_tokens":17012,"output_tokens":27,"input_tokens_details":{"cached_tokens":16000}}`)

	inputTokens, outputTokens, cachedTokens := extractOpenAIUsage(usage)

	if inputTokens != 17012 {
		t.Fatalf("expected input_tokens %d, got %d", 17012, inputTokens)
	}
	if outputTokens != 27 {
		t.Fatalf("expected output_tokens %d, got %d", 27, outputTokens)
	}
	if cachedTokens != 16000 {
		t.Fatalf("expected cached_tokens %d, got %d", 16000, cachedTokens)
	}
}

func TestConvertOpenAIResponseToClaudeNonStream_MapsResponsesStyleCachedUsage(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	response := []byte(`{
		"id":"resp_cached",
		"object":"response",
		"created":1770000000,
		"model":"gpt-5.4-high",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"ok"},
			"finish_reason":"stop"
		}],
		"usage":{
			"input_tokens":17012,
			"output_tokens":27,
			"total_tokens":17039,
			"input_tokens_details":{"cached_tokens":16000}
		}
	}`)

	out := ConvertOpenAIResponseToClaudeNonStream(ctx, "", originalRequest, nil, response, nil)
	parsed := gjson.ParseBytes(out)

	if got := parsed.Get("usage.input_tokens").Int(); got != 17012 {
		t.Fatalf("expected usage.input_tokens %d, got %d", 17012, got)
	}
	if got := parsed.Get("usage.output_tokens").Int(); got != 27 {
		t.Fatalf("expected usage.output_tokens %d, got %d", 27, got)
	}
	if got := parsed.Get("usage.cache_read_input_tokens").Int(); got != 16000 {
		t.Fatalf("expected usage.cache_read_input_tokens %d, got %d", 16000, got)
	}
}
