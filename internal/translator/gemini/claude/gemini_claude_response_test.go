package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertGeminiResponseToClaude_StreamMessageStartUsesGPTOriginalRequestFallback(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"Count the words in this sentence."}]}`)
	requestRawJSON := []byte(`{"contents":[]}`)
	rawJSON := []byte(`{"responseId":"resp_123","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"text":"hello"}]}}]}`)

	var param any
	outputs := ConvertGeminiResponseToClaude(ctx, "", originalRequest, requestRawJSON, rawJSON, &param)

	messageStart := findAnthropicEvent(outputs, "message_start")
	if !messageStart.Exists() {
		t.Fatal("expected message_start event")
	}
	if got := messageStart.Get("message.usage.input_tokens").Int(); got <= 0 {
		t.Fatalf("expected fallback input_tokens > 0, got %d", got)
	}
	if got := messageStart.Get("message.usage.output_tokens").Int(); got != 1 {
		t.Fatalf("expected message_start output_tokens %d, got %d", 1, got)
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
