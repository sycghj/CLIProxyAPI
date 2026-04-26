package claude

import (
	"context"
	"strings"
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

func TestConvertOpenAIResponseToClaude_StreamMessageStartEstimatesInputTokensForGPTModel(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"stream":true,"messages":[]}`)
	requestRawJSON := []byte(`{
		"model":"gpt-4o",
		"stream":true,
		"messages":[
			{"role":"system","content":"You are a careful assistant."},
			{"role":"user","content":"Count the words in this sentence."}
		]
	}`)
	chunk := []byte("data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"created\":1770000000,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Sure\"}}]}")

	var param any
	outputs := ConvertOpenAIResponseToClaude(ctx, "", originalRequest, requestRawJSON, chunk, &param)

	messageStart := findAnthropicEvent(outputs, "message_start")
	if !messageStart.Exists() {
		t.Fatal("expected message_start event")
	}
	if got := messageStart.Get("message.usage.input_tokens").Int(); got <= 0 {
		t.Fatalf("expected estimated non-zero message_start input_tokens, got %d", got)
	}
	if got := messageStart.Get("message.usage.output_tokens").Int(); got != 1 {
		t.Fatalf("expected message_start output_tokens %d, got %d", 1, got)
	}
}

func TestConvertOpenAIResponseToClaude_StreamMessageDeltaKeepsRealUsage(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"stream":true,"messages":[]}`)
	requestRawJSON := []byte(`{
		"model":"gpt-5.4-high",
		"stream":true,
		"messages":[
			{"role":"user","content":"hello"}
		]
	}`)
	chunks := [][]byte{
		[]byte("data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-5.4-high\",\"created\":1770000000,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}"),
		[]byte("data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-5.4-high\",\"created\":1770000000,\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}"),
		[]byte("data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-5.4-high\",\"created\":1770000000,\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":9,\"total_tokens\":51}}"),
	}

	var param any
	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertOpenAIResponseToClaude(ctx, "", originalRequest, requestRawJSON, chunk, &param)...)
	}

	messageDelta := findAnthropicEvent(outputs, "message_delta")
	if !messageDelta.Exists() {
		t.Fatal("expected message_delta event")
	}
	if got := messageDelta.Get("usage.input_tokens").Int(); got != 42 {
		t.Fatalf("expected final message_delta input_tokens %d, got %d", 42, got)
	}
	if got := messageDelta.Get("usage.output_tokens").Int(); got != 9 {
		t.Fatalf("expected final message_delta output_tokens %d, got %d", 9, got)
	}
}

func findAnthropicEvent(outputs [][]byte, eventType string) gjson.Result {
	for _, out := range outputs {
		for _, line := range strings.Split(string(out), "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := gjson.Parse(strings.TrimPrefix(line, "data: "))
			if payload.Get("type").String() == eventType {
				return payload
			}
		}
	}
	return gjson.Result{}
}
