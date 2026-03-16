use super::*;
use std::sync::Arc;

/// Context that can be passed to tools.
/// Use `Arc::new(your_context) as Context` to create a context,
/// then use `context.downcast_ref::<YourType>()` in tools to access it.
pub type Context = Arc<dyn std::any::Any + Send + Sync>;

/// Trait for tools that require context.
/// Implement this trait in addition to `Tool` to receive context during tool execution.
///
/// # Example
/// ```ignore
/// use rig::tool::{Tool, ToolWithContext, Context};
///
/// struct MyTool;
///
/// impl Tool for MyTool {
///     const NAME: &'static str = "my_tool";
///     type Error = MyError;
///     type Args = MyArgs;
///     type Output = String;
///
///     async fn definition(&self, _prompt: String) -> ToolDefinition {
///         // ... definition
///     }
///
///     async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
///         Err(MyError::ContextRequired) // Fallback
///     }
/// }
///
/// impl ToolWithContext for MyTool {
///     async fn call_with_context(
///         &self,
///         args: Self::Args,
///         context: &Context,
///     ) -> Result<Self::Output, Self::Error> {
///         let my_ctx = context.downcast_ref::<MyContextType>()?;
///         // Use context here
///         Ok(my_ctx.process(args))
///     }
/// }
///
/// // Add to agent using tool_with_context()
/// let agent = client
///     .agent(MODEL)
///     .tool_with_context(MyTool::new())
///     .build();
///
/// // Use context when calling the agent
/// agent.prompt("...")
///     .with_context(Arc::new(my_context))
///     .await?;
/// ```
pub trait ToolWithContext: Tool {
    /// Execute the tool with context.
    fn call_with_context(
        &self,
        args: Self::Args,
        context: &Context,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

/// Internal wrapper for tools that implement ToolWithContext.
/// This wrapper properly routes context-aware calls.
///
/// Note: This does NOT implement `Tool` to avoid conflicts with the blanket `ToolDyn` impl.
/// It only implements `ToolDyn` directly with the proper context routing.
pub(crate) struct ContextAwareTool<T: Tool + ToolWithContext>(pub(crate) T);

impl<T: Tool + ToolWithContext + 'static> ToolDyn for ContextAwareTool<T> {
    fn name(&self) -> String {
        self.0.name()
    }

    fn definition<'a>(&'a self, prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(self.0.definition(prompt))
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            match serde_json::from_str(&args) {
                Ok(args) => <T as Tool>::call(&self.0, args)
                    .await
                    .map_err(|e| ToolError::ToolCallError(Box::new(e)))
                    .and_then(|output| {
                        serde_json::to_string(&output).map_err(ToolError::JsonError)
                    }),
                Err(e) => Err(ToolError::JsonError(e)),
            }
        })
    }

    fn call_with_context<'a>(
        &'a self,
        args: String,
        context: &'a Context,
    ) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            tracing::debug!(target: "rig", "Using ToolWithContext for {}", T::NAME);
            match serde_json::from_str(&args) {
                Ok(args) => <T as ToolWithContext>::call_with_context(&self.0, args, context)
                    .await
                    .map_err(|e| ToolError::ToolCallError(Box::new(e)))
                    .and_then(|output| {
                        serde_json::to_string(&output).map_err(ToolError::JsonError)
                    }),
                Err(e) => Err(ToolError::JsonError(e)),
            }
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        &self.0
    }
}
