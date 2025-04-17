import {Hono} from 'hono';
import {streamSSE} from 'hono/streaming';
import OpenAI from 'openai';
import type {ChatCompletionCreateParams, EmbeddingCreateParams} from 'openai/resources';

import {getApiKey, withRetry, withoutBalancing} from '../utils/apikey.ts';
import {createErrorResponse, createHonoErrorResponse} from '../utils/error';
import {openaiAuthMiddleware} from '../utils/middleware'

const oai = new Hono();

oai.use('/*', openaiAuthMiddleware);

const baseURL = "https://generativelanguage.googleapis.com/v1beta/openai/";

// 创建聊天
oai.post('/chat/completions', async (c) => {
    const {messages, model: originalModel, tools: originalTools, tool_choice, stream = false} = await c.req.json() as ChatCompletionCreateParams & {
        stream?: boolean
    };

    try {
        // --- Google Search Tool Injection Logic ---
        let actualModelId = originalModel;
        let finalTools = originalTools; // Initialize with original tools
        const isSearchModel = originalModel.endsWith('-search');

        if (isSearchModel) {
            actualModelId = originalModel.replace('-search', '');
            console.log(`Search model detected. Using base model: ${actualModelId}`);
            const googleSearchTool = { type: "function", function: { name: "googleSearch", description: "Google Search" } }; // Correct format for OpenAI SDK tools

            if (finalTools && Array.isArray(finalTools)) {
                 // Check if googleSearchTool already exists to avoid duplicates (optional but good practice)
                if (!finalTools.some(tool => tool.type === 'function' && tool.function.name === 'googleSearch')) {
                    finalTools = [...finalTools, googleSearchTool];
                }
            } else {
                finalTools = [googleSearchTool];
            }
            console.log('Injecting Google Search tool.');
        }
        // --- End Google Search Tool Injection Logic ---

        // 处理流式响应
        if (stream) {
            // 创建流式请求
            return streamSSE(c, async (stream) => {
                try {
                    const completion = await withRetry(actualModelId, async (key) => { // Use actualModelId for retry logic
                        const openai = new OpenAI({
                            apiKey: key, baseURL: baseURL
                        });

                        return await openai.chat.completions.create({
                            model: actualModelId, // Use actualModelId
                            messages: messages,
                            tools: finalTools, // Use finalTools
                            tool_choice: tool_choice,
                            stream: true,
                        });
                    });

                    for await (const chunk of completion) {
                        await stream.writeSSE({
                            data: JSON.stringify(chunk)
                        });
                    }
                    await stream.writeSSE({data: '[DONE]'});
                } catch (error) {
                    console.error('流式处理错误:', error);
                    const {body} = createErrorResponse(error);
                    await stream.writeSSE({
                        data: JSON.stringify(body)
                    });
                }
            });
        }

        // 非流式响应
        const response = await withRetry(actualModelId, async (key) => { // Use actualModelId for retry logic
            const openai = new OpenAI({
                apiKey: key, baseURL: baseURL
            });

            return await openai.chat.completions.create({
                 model: actualModelId, // Use actualModelId
                 messages: messages,
                 tools: finalTools, // Use finalTools
                 tool_choice: tool_choice,
            });
        });

        return c.json(response);
    } catch (error: any) {
        console.error('API调用错误:', error);
        return createHonoErrorResponse(c, error);
    }
})
// 列出模型
oai.get('/models', async (c) => {
    try {
        // 1. Fetch original models from upstream
        const originalModelsResponse = await withoutBalancing(async (key) => {
            const openai = new OpenAI({
                apiKey: key, baseURL: baseURL
            });
            return await openai.models.list();
        });

        const originalModelsData = originalModelsResponse.data || [];

        // 2. Identify models eligible for -search suffix and create variants
        const searchModelsData = originalModelsData
            .filter(model =>
                // Apply the rule: gemini-2.x or higher series
                /^gemini-[2-9]\.\d/.test(model.id) &&
                // Ensure it's not already a search model (though unlikely from upstream)
                !model.id.endsWith('-search')
            )
            .map(model => ({
                ...model, // Copy original properties
                id: `${model.id}-search`, // Append -search
                // Ensure 'created' and 'owned_by' are present if needed, copying from original
                created: model.created || Math.floor(Date.now() / 1000),
                owned_by: model.owned_by || "google",
            }));

        // 3. Combine original and search models
        const combinedModelsData = [...originalModelsData, ...searchModelsData];

        // 4. Return the combined list
        return c.json({
            object: "list", data: combinedModelsData
        });
    } catch (error: any) {
        console.error('获取模型错误:', error);
        return createHonoErrorResponse(c, error);
    }
})


// 检索模型
oai.get('/models/:model', async (c) => {
    const {model: modelId} = c.req.param();
    
    try {
        const model = await withoutBalancing(async (key) => {
            const openai = new OpenAI({
                apiKey: key, baseURL: baseURL
            });
            return await openai.models.retrieve(modelId);
        });
        
        return c.json(model);
    } catch (error: any) {
        console.error('检索模型错误:', error);
        return createHonoErrorResponse(c, error);
    }
})

// Embeddings
oai.post('/embeddings', async (c) => {
    const {model, input, encoding_format, dimensions} = await c.req.json() as EmbeddingCreateParams;

    if (!model || !input) {
        return createHonoErrorResponse(c, {
            message: "请求体必须包含 'model' 和 'input' 参数。", type: "invalid_request_error", status: 400
        });
    }

    try {
        const embeddingResponse = await withRetry(model, async (key) => {
            const openai = new OpenAI({
                apiKey: key, baseURL: baseURL
            });
            
            return await openai.embeddings.create({
                model: model,
                input: input, 
                ...(encoding_format && {encoding_format: encoding_format}), 
                ...(dimensions && {dimensions: dimensions})
            });
        });

        return c.json(embeddingResponse);
    } catch (error: any) {
        console.error('创建 Embeddings 时出错:', error);
        return createHonoErrorResponse(c, error);
    }
});

export default oai;