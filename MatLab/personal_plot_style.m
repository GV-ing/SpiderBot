function h = personal_plot_style(x, Y, x_label_latex, y_label_latex, legend_labels, legend_position, pdf_name)
    % Impostazioni stile richiesto
    set(0, 'DefaultTextInterpreter', 'latex')
    set(0, 'DefaultLegendInterpreter', 'latex')
    set(0, 'DefaultAxesTickLabelInterpreter', 'latex')
    lw = 2;

    % Figure generation
    h = figure('Renderer', 'painters', 'Position', [10 10 900 350]);

    % Colori e stili (Estesi per 3 segnali: Rosso, Verde, Blu)
    colors = {[1, 0, 0], [0, 0.5, 0], [0, 0, 1]}; % RGB
    styles = {'-', '-', '-'}; 

    hold on;
    for i = 1:size(Y, 2)
        plot(x, Y(:, i), styles{i}, 'Linewidth', lw, 'Color', colors{i});
    end

    % Legend
    legend(legend_labels);
    legend('Location', legend_position, 'Orientation', 'horizontal', 'AutoUpdate', 'off')

    % Labels 
    xlabel(x_label_latex)
    ylabel(y_label_latex)
    set(gca, 'FontSize', 22);

    % Grid
    grid on
    box on

    % Options
    set(gcf, 'color', 'w');
    set(h, 'MenuBar', 'none');
    set(h, 'ToolBar', 'none');

    % Limits
    xlim([x(1) x(end)])
    ylim_lb = min(Y(:));
    ylim_ub = max(Y(:));
    ylim([ylim_lb ylim_ub]);

    % Fix inner position (come richiesto)
    set(gca, 'InnerPosition', [0.1400 0.32 0.82 0.55])
    annotation('rectangle', [0 0 1 1], 'Color', 'w');

    exportgraphics(h, pdf_name);
end