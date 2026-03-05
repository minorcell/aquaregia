use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::{
    Expr, FnArg, ItemFn, Lit, LitStr, Meta, Pat, Token, punctuated::Punctuated, parse_macro_input,
};

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let metas = match Punctuated::<Meta, Token![,]>::parse_terminated.parse(attr) {
        Ok(metas) => metas,
        Err(err) => return err.to_compile_error().into(),
    };

    let mut description: Option<LitStr> = None;
    for meta in metas {
        match meta {
            Meta::NameValue(name_value) if name_value.path.is_ident("description") => {
                if description.is_some() {
                    return syn::Error::new(
                        name_value.span(),
                        "duplicate `description` argument in #[tool(...)]",
                    )
                    .to_compile_error()
                    .into();
                }
                match name_value.value {
                    Expr::Lit(expr_lit) => match expr_lit.lit {
                        Lit::Str(lit) => description = Some(lit),
                        _ => {
                            return syn::Error::new(
                                expr_lit.span(),
                                "`description` must be a string literal",
                            )
                            .to_compile_error()
                            .into();
                        }
                    },
                    _ => {
                        return syn::Error::new(
                            name_value.value.span(),
                            "`description` must be a string literal",
                        )
                        .to_compile_error()
                        .into();
                    }
                }
            }
            other => {
                return syn::Error::new(
                    other.span(),
                    "unsupported #[tool(...)] argument; expected `description = \"...\"`",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    let input = parse_macro_input!(item as ItemFn);
    if input.sig.asyncness.is_none() {
        return syn::Error::new(
            input.sig.fn_token.span(),
            "#[tool] requires an `async fn` handler",
        )
        .to_compile_error()
        .into();
    }
    if !input.sig.generics.params.is_empty() || input.sig.generics.where_clause.is_some() {
        return syn::Error::new(
            input.sig.generics.span(),
            "#[tool] does not support generic parameters yet",
        )
        .to_compile_error()
        .into();
    }

    let vis = input.vis;
    let attrs = input.attrs;
    let fn_name = input.sig.ident;
    let output = input.sig.output;
    let body = input.block;

    let mut arg_idents = Vec::new();
    let mut arg_tys = Vec::new();
    for arg in input.sig.inputs {
        match arg {
            FnArg::Receiver(receiver) => {
                return syn::Error::new(
                    receiver.span(),
                    "#[tool] does not support methods with `self`",
                )
                .to_compile_error()
                .into();
            }
            FnArg::Typed(pat_type) => {
                let ident = match *pat_type.pat {
                    Pat::Ident(pat_ident)
                        if pat_ident.by_ref.is_none()
                            && pat_ident.mutability.is_none()
                            && pat_ident.subpat.is_none() =>
                    {
                        pat_ident.ident
                    }
                    other => {
                        return syn::Error::new(
                            other.span(),
                            "#[tool] parameters must be simple identifiers, e.g. `city: String`",
                        )
                        .to_compile_error()
                        .into();
                    }
                };
                arg_idents.push(ident);
                arg_tys.push(*pat_type.ty);
            }
        }
    }

    let description_lit =
        description.unwrap_or_else(|| LitStr::new("", proc_macro2::Span::call_site()));
    let args_ident = format_ident!("__AquaregiaToolArgs_{}", fn_name);
    let handler_ident = format_ident!("__aquaregia_tool_handler_{}", fn_name);
    let arg_extracts = arg_idents.iter().map(|ident| quote! { args.#ident });

    quote! {
        #(#attrs)*
        #vis fn #fn_name() -> ::aquaregia::Tool {
            #[allow(non_camel_case_types)]
            #[derive(::aquaregia::__aquaregia_serde::Deserialize, ::aquaregia::__aquaregia_schemars::JsonSchema)]
            struct #args_ident {
                #( #arg_idents: #arg_tys, )*
            }

            ::aquaregia::tool(stringify!(#fn_name))
                .description(#description_lit)
                .execute(|args: #args_ident| async move {
                    #handler_ident( #( #arg_extracts ),* )
                        .await
                        .map_err(|err| ::aquaregia::ToolExecError::Execution(err.to_string()))
                })
        }

        async fn #handler_ident( #( #arg_idents: #arg_tys ),* ) #output {
            #body
        }
    }
    .into()
}
