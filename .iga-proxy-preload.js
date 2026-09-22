// Preload: convert axios old-style HTTP proxy forwarding to CONNECT tunneling.
//
// The @iga-pages CLI bundles axios, which — when an HTTPS_PROXY env var is
// set to http://... — sends HTTPS requests using "absolute URI" form:
//     GET https://open.volcengineapi.com/... HTTP/1.1
//     Host: 127.0.0.1:18080   (the proxy)
// The upstream proxy then forwards this as plain HTTP to port 443, causing
// 400 "The plain HTTP request was sent to HTTPS port".
//
// This preload intercepts http.request() calls whose path starts with
// "https://" (the old-style proxy signature) and converts them into
// https.request() calls using HttpsProxyAgent, which correctly issues:
//     CONNECT open.volcengineapi.com:443 HTTP/1.1
// to the proxy before performing the TLS handshake.

const http = require('http');
const https = require('https');

const proxy =
    process.env.HTTPS_PROXY || process.env.https_proxy ||
    process.env.HTTP_PROXY  || process.env.http_proxy;

if (proxy) {
    let HttpsProxyAgent = null;
    try {
        ({ HttpsProxyAgent } = require('https-proxy-agent'));
    } catch (e) {
        // Package not available — CLI will fail with its original error.
    }

    if (HttpsProxyAgent) {

    const httpsAgent = new HttpsProxyAgent(proxy);
    const origHttpRequest = http.request;
    const origHttpsRequest = https.request;

    http.request = function (options, callback) {
        // Detect old-style proxy forwarding: path is a full https:// URL.
        const path = typeof options === 'object' ? options.path : null;
        if (path && typeof path === 'string' && path.startsWith('https://')) {
            try {
                const parsed = new URL(path);
                const newOptions = {
                    ...options,
                    protocol: 'https:',
                    hostname: parsed.hostname,
                    host: parsed.host,
                    port: parsed.port || 443,
                    path: parsed.pathname + parsed.search,
                    agent: httpsAgent,
                    // Clear proxy-specific fields set by axios setProxy()
                    headers: { ...options.headers, Host: parsed.host },
                };
                return origHttpsRequest.call(https, newOptions, callback);
            } catch (e) {
                // URL parse failed — fall through to original behavior.
            }
        }
        return origHttpRequest.call(http, options, callback);
    };

    // Also patch https.request in case axios calls it directly with proxy opts.
    https.request = function (options, callback) {
        const path = typeof options === 'object' ? options.path : null;
        if (path && typeof path === 'string' && path.startsWith('https://')) {
            try {
                const parsed = new URL(path);
                const newOptions = {
                    ...options,
                    protocol: 'https:',
                    hostname: parsed.hostname,
                    host: parsed.host,
                    port: parsed.port || 443,
                    path: parsed.pathname + parsed.search,
                    agent: httpsAgent,
                    headers: { ...options.headers, Host: parsed.host },
                };
                return origHttpsRequest.call(https, newOptions, callback);
            } catch (e) {
                // fall through
            }
        }
        // For normal https requests without proxy path, still use the agent
        if (typeof options === 'object' && !options.agent &&
            (!options.path || !options.path.startsWith('http'))) {
            options = { ...options, agent: httpsAgent };
        }
        return origHttpsRequest.call(https, options, callback);
    };
    } // end if (HttpsProxyAgent)
}
