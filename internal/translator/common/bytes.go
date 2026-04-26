package common

import (
	"strconv"
	"strings"

	executorhelps "github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func WrapGeminiCLIResponse(response []byte) []byte {
	out, err := sjson.SetRawBytes([]byte(`{"response":{}}`), "response", response)
	if err != nil {
		return response
	}
	return out
}

func GeminiTokenCountJSON(count int64) []byte {
	out := make([]byte, 0, 96)
	out = append(out, `{"totalTokens":`...)
	out = strconv.AppendInt(out, count, 10)
	out = append(out, `,"promptTokensDetails":[{"modality":"TEXT","tokenCount":`...)
	out = strconv.AppendInt(out, count, 10)
	out = append(out, `}]}`...)
	return out
}

func ClaudeInputTokensJSON(count int64) []byte {
	out := make([]byte, 0, 32)
	out = append(out, `{"input_tokens":`...)
	out = strconv.AppendInt(out, count, 10)
	out = append(out, '}')
	return out
}

func ApplyInitialClaudeMessageUsage(template []byte, requestRawJSON, originalRequestRawJSON []byte, responseModels ...string) []byte {
	if gjson.GetBytes(template, "message.usage.input_tokens").Int() != 0 || gjson.GetBytes(template, "message.usage.output_tokens").Int() != 0 {
		return template
	}

	model := firstGPTModel(requestRawJSON, originalRequestRawJSON, responseModels...)
	if model == "" {
		return template
	}

	inputTokens := EstimateInitialOpenAIChatTokens(model, requestRawJSON, originalRequestRawJSON)
	template, _ = sjson.SetBytes(template, "message.usage.input_tokens", inputTokens)
	template, _ = sjson.SetBytes(template, "message.usage.output_tokens", 1)
	return template
}

func EstimateInitialOpenAIChatTokens(model string, payloads ...[]byte) int64 {
	enc, err := executorhelps.TokenizerForModel(model)
	if err != nil {
		return 1
	}

	for _, payload := range payloads {
		count, err := executorhelps.CountOpenAIChatTokens(enc, payload)
		if err == nil && count > 0 {
			return count
		}
	}
	return 1
}

func firstGPTModel(requestRawJSON, originalRequestRawJSON []byte, responseModels ...string) string {
	for _, model := range append([]string{
		gjson.GetBytes(requestRawJSON, "model").String(),
		gjson.GetBytes(originalRequestRawJSON, "model").String(),
	}, responseModels...) {
		model = strings.ToLower(strings.TrimSpace(model))
		if strings.HasPrefix(model, "gpt-") {
			return model
		}
	}
	return ""
}

func SSEEventData(event string, payload []byte) []byte {
	out := make([]byte, 0, len(event)+len(payload)+14)
	out = append(out, "event: "...)
	out = append(out, event...)
	out = append(out, '\n')
	out = append(out, "data: "...)
	out = append(out, payload...)
	return out
}

func AppendSSEEventString(out []byte, event, payload string, trailingNewlines int) []byte {
	out = append(out, "event: "...)
	out = append(out, event...)
	out = append(out, '\n')
	out = append(out, "data: "...)
	out = append(out, payload...)
	for i := 0; i < trailingNewlines; i++ {
		out = append(out, '\n')
	}
	return out
}

func AppendSSEEventBytes(out []byte, event string, payload []byte, trailingNewlines int) []byte {
	out = append(out, "event: "...)
	out = append(out, event...)
	out = append(out, '\n')
	out = append(out, "data: "...)
	out = append(out, payload...)
	for i := 0; i < trailingNewlines; i++ {
		out = append(out, '\n')
	}
	return out
}
