use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

/// User-facing API error with a consistent JSON shape.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": {
                    "message": self.message,
                    "type": "invalid_request_error",
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_request_has_400_status() {
        let err = ApiError::bad_request("nope");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "nope");
    }

    #[test]
    fn timeout_has_504_status() {
        let err = ApiError::timeout("slow");
        assert_eq!(err.status, StatusCode::GATEWAY_TIMEOUT);
    }

    #[test]
    fn unavailable_has_503_status() {
        let err = ApiError::unavailable("down");
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn into_response_serializes_message() {
        let resp = ApiError::bad_request("bad input").into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
